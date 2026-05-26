#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모기 비행 궤적 예측 v4 - 정준 좌표계 + 데이터 증강 + 앙상블
==============================================================

핵심 혁신:
  1. 정준(Canonical) 좌표계 변환
     - 모든 궤적을 '마지막 속도 방향 = +x축'으로 회전 정규화
     - 회전 불변 피처 → 방향과 무관하게 동일 패턴 학습
  2. 데이터 증강 (×10)
     - x축 주변 무작위 회전으로 훈련 샘플 10배 확장
     - MLP는 100,000개 표본으로 학습 (기존 10,000개 대비 10배)
  3. XGBoost + LightGBM + MLP 앙상블 (OOF 가중치 최적화)

개발 환경:
  Python: 3.10.x  torch: 2.8.0  numpy: 2.0.2
  pandas: 2.3.3  scikit-learn: 1.6.1
  xgboost: 설치 버전  lightgbm: 설치 버전  tqdm: 4.67.3
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ============================================================
# 경로 설정 (상대 경로)
# ============================================================
DATA_DIR  = Path("./open")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
LABEL_CSV = DATA_DIR / "train_labels.csv"
SUBM_CSV  = DATA_DIR / "sample_submission.csv"
OUT_CSV   = Path("./submission.csv")

SEED    = 42
N_FOLDS = 5
BATCH   = 1024          # 증강 데이터(100k)에 맞게 배치 확대
EPOCHS  = 300
LR      = 1e-3
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_AUG   = 9            # 증강 배수: 1+9=10배 훈련 데이터

POLY_DEG  = 2
FUTURE_MS = 80.0
T_IN      = np.arange(-400, 1, 40, dtype=float)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 평가 지표
# ============================================================
def r_hit(pred: np.ndarray, true: np.ndarray, thr: float = 0.01) -> float:
    dist = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    return float(np.mean(dist <= thr))


# ============================================================
# 데이터 로딩
# ============================================================
def load_trajectories(directory: Path, ids: np.ndarray, desc: str = "") -> np.ndarray:
    seqs = []
    for id_ in tqdm(ids, desc=desc, ncols=80):
        df = pd.read_csv(directory / f"{id_}.csv")
        seqs.append(df[["x", "y", "z"]].values.astype(np.float32))
    return np.stack(seqs)


# ============================================================
# 정준 좌표계 변환
# ============================================================
def _rotation_to_x(vel: np.ndarray) -> np.ndarray:
    """
    vel (3,) 을 +x 축으로 정렬하는 3×3 회전 행렬 반환 (Rodrigues 공식)
    """
    v_norm = np.linalg.norm(vel)
    if v_norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    v = vel / v_norm
    x = np.array([1.0, 0.0, 0.0])
    cos_t = float(np.dot(v, x))
    cross = np.cross(v, x)
    sin_t = np.linalg.norm(cross)
    if sin_t < 1e-8:
        return np.eye(3, dtype=np.float32) if cos_t > 0 else np.diag([-1., 1., -1.]).astype(np.float32)
    k = cross / sin_t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    R = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)
    return R.astype(np.float32)


def canonical_normalize_batch(trajs: np.ndarray):
    """
    (N, 11, 3) → canonical trajectories, rotation matrices, centers

    반환:
      X_can:   (N, 11, 3) — 정준화된 궤적 (마지막 포인트 = 원점, 마지막 속도 = +x)
      R_all:   (N, 3, 3)  — 회전 행렬 (원본 → 정준)
      centers: (N, 3)     — 각 샘플의 마지막 위치 (평행이동 기준)
    """
    N = len(trajs)
    centers = trajs[:, -1, :].copy()                    # (N, 3)
    X_can   = np.zeros_like(trajs)                      # (N, 11, 3)
    R_all   = np.zeros((N, 3, 3), dtype=np.float32)

    for i in range(N):
        center = centers[i]
        cen    = trajs[i] - center                      # (11, 3), last=(0,0,0)
        vel    = cen[-1] - cen[-2]                      # 마지막 속도 벡터
        R      = _rotation_to_x(vel)
        X_can[i]  = cen @ R.T                           # 회전 적용
        R_all[i]  = R

    return X_can.astype(np.float32), R_all, centers.astype(np.float32)


def canonical_to_original(pred_can: np.ndarray, R_all: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """정준 예측 → 원본 좌표 복원: pred_orig[i] = R[i]^T @ pred_can[i] + center[i]"""
    # einsum 'nji,nj->ni' = batched R.T @ v
    return np.einsum('nji,nj->ni', R_all, pred_can) + centers


# ============================================================
# 정준 프레임에서 다항식 외삽
# ============================================================
def poly_extrap_can_batch(X_can: np.ndarray, deg: int = POLY_DEG,
                          t_future: float = FUTURE_MS) -> np.ndarray:
    """
    X_can: (N, 11, 3) — 정준 궤적 (마지막 포인트 = 원점)
    반환: (N, 3) 정준 프레임 다항식 예측 (원점에서의 변위)
    """
    N = len(X_can)
    preds = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        for i in range(3):
            coef = np.polyfit(T_IN, X_can[n, :, i], deg)
            preds[n, i] = float(np.polyval(coef, t_future))
    return preds


# ============================================================
# 피처 추출 (정준 프레임, 196차원)
# ============================================================
def build_features(traj_can: np.ndarray, poly_can: np.ndarray) -> np.ndarray:
    """
    정준 궤적 기반 196차원 피처. 마지막 포인트 = 원점이므로
    centered = traj_can 과 동일.
    """
    eps        = 1e-8
    cen        = traj_can.astype(np.float32)              # (11, 3), last=(0,0,0)
    vel        = np.diff(cen, axis=0)                     # (10, 3)
    acc        = np.diff(vel, axis=0)                     # ( 9, 3)
    jerk       = np.diff(acc, axis=0)                     # ( 8, 3)
    spd        = np.linalg.norm(vel, axis=-1, keepdims=True)
    direc      = vel / (spd + eps)
    curv_cross = np.cross(vel[:-1], acc)                  # ( 9, 3)
    curv_mag   = np.linalg.norm(curv_cross, axis=-1)      # ( 9,)
    poly_r     = poly_can.astype(np.float32)              # 원점 기준 변위

    local_lin  = (2.0 * vel[-1]).astype(np.float32)

    local_poly5 = np.empty(3, dtype=np.float32)
    for i in range(3):
        c = np.polyfit(T_IN[-5:], cen[-5:, i], 2)
        local_poly5[i] = float(np.polyval(c, FUTURE_MS))

    loo = np.empty(6, dtype=np.float32)
    for i in range(3):
        c = np.polyfit(T_IN[:-2], cen[:-2, i], POLY_DEG)
        loo[i*2]   = float(np.polyval(c, -40)) - cen[-2, i]
        loo[i*2+1] = float(np.polyval(c,   0)) - cen[-1, i]  # cen[-1]=0

    return np.concatenate([
        cen.ravel(), vel.ravel(), acc.ravel(), jerk.ravel(),
        direc.ravel(), curv_cross.ravel(), curv_mag,
        poly_r, [spd[-1, 0]],
        local_lin, local_poly5, loo,
    ]).astype(np.float32)   # 196


def extract_all_flat(X_can: np.ndarray, poly_can: np.ndarray) -> np.ndarray:
    return np.stack([build_features(X_can[i], poly_can[i])
                     for i in range(len(X_can))]).astype(np.float32)


# ============================================================
# 데이터 증강 (x축 회전, MLP 전용)
# ============================================================
def augment_around_x(X_can: np.ndarray, y_can: np.ndarray, poly_can: np.ndarray,
                     n_aug: int = N_AUG) -> tuple:
    """
    x축 주변 무작위 회전으로 (1+n_aug)배 데이터 생성.
    last velocity(+x축 방향)를 유지하면서 y-z 평면을 회전.
    """
    N = len(X_can)
    all_X, all_y, all_p = [X_can], [y_can], [poly_can]

    for _ in range(n_aug):
        th   = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
        c, s = np.cos(th), np.sin(th)

        Rx = np.zeros((N, 3, 3), dtype=np.float32)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = c;   Rx[:, 1, 2] = -s
        Rx[:, 2, 1] = s;   Rx[:, 2, 2] = c

        # X_can: (N,11,3) → (N,11,3)
        X_aug = np.einsum('nij,ntj->nti', Rx, X_can)
        y_aug = np.einsum('nij,nj->ni',   Rx, y_can)
        p_aug = np.einsum('nij,nj->ni',   Rx, poly_can)

        all_X.append(X_aug)
        all_y.append(y_aug)
        all_p.append(p_aug)

    return (np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
            np.concatenate(all_p, axis=0))


# ============================================================
# XGBoost
# ============================================================
def train_xgb_fold(X_tr, y_can_tr, X_val, y_can_val):
    models, val_preds = [], np.zeros((len(X_val), 3))
    for axis in range(3):
        m = xgb.XGBRegressor(
            n_estimators=2000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=50, random_state=SEED,
            n_jobs=-1, verbosity=0,
        )
        m.fit(X_tr, y_can_tr[:, axis],
              eval_set=[(X_val, y_can_val[:, axis])], verbose=False)
        val_preds[:, axis] = m.predict(X_val)
        models.append(m)
    return models, val_preds


# ============================================================
# LightGBM
# ============================================================
def train_lgb_fold(X_tr, y_can_tr, X_val, y_can_val):
    models, val_preds = [], np.zeros((len(X_val), 3))
    for axis in range(3):
        m = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=63, learning_rate=0.03,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr, y_can_tr[:, axis],
              eval_set=[(X_val, y_can_val[:, axis])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=0)])
        val_preds[:, axis] = m.predict(X_val)
        models.append(m)
    return models, val_preds


# ============================================================
# MLP (증강 데이터로 훈련)
# ============================================================
class TrajDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int = 196):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 512),    nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128),                       nn.GELU(),
            nn.Linear(128, 3),
        )
    def forward(self, x): return self.net(x)


def train_mlp_fold(X_can_tr, y_can_tr, poly_can_tr,
                   X_can_val, poly_can_val, R_val, centers_val, y_abs_val):
    """
    증강 데이터로 MLP 훈련.
    검증은 원본(비증강) 검증 세트에서 R-Hit@1cm로 평가.
    """
    PATIENCE = 40

    # 훈련 데이터 증강 (10배)
    X_aug, y_aug, p_aug = augment_around_x(X_can_tr, y_can_tr, poly_can_tr)
    print(f"    MLP 증강 훈련 샘플: {len(X_aug):,}", flush=True)

    # 피처 추출
    Xf_tr = extract_all_flat(X_aug, p_aug)              # (10N, 196)
    Xf_val = extract_all_flat(X_can_val, poly_can_val)  # (N_val, 196)

    sc   = StandardScaler().fit(Xf_tr)
    Xts  = sc.transform(Xf_tr).astype(np.float32)
    Xvs  = sc.transform(Xf_val).astype(np.float32)
    Xvt  = torch.from_numpy(Xvs).to(DEVICE)

    dl    = DataLoader(TrajDataset(Xts, y_aug.astype(np.float32)),
                       batch_size=BATCH, shuffle=True, num_workers=0)
    model = ResidualMLP(in_dim=Xts.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_hit, best_state, no_imp = -1.0, None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                can_corr = model(Xvt).cpu().numpy()          # (N_val, 3) canonical residual
            pred_can  = poly_can_val + can_corr
            pred_orig = canonical_to_original(pred_can, R_val, centers_val)
            hit = r_hit(pred_orig, y_abs_val)
            if hit > best_hit:
                best_hit  = hit
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 10
            if no_imp >= PATIENCE:
                print(f"    [MLP] 조기종료 ep={ep:3d} best={best_hit:.5f}", flush=True)
                break
        if ep % 100 == 0:
            print(f"    [MLP] ep={ep:3d} | best={best_hit:.5f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        can_corr = model(Xvt).cpu().numpy()
    pred_can  = poly_can_val + can_corr
    pred_orig = canonical_to_original(pred_can, R_val, centers_val)
    return model, sc, pred_orig, best_hit


# ============================================================
# 앙상블 가중치 최적화
# ============================================================
def optimize_weights(oof_xgb, oof_lgbm, oof_mlp, y_abs):
    best_hit, best_w = -1.0, (1/3, 1/3, 1/3)
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 10)
            if w3 < -1e-9: continue
            w3 = max(0.0, w3)
            pred = w1*oof_xgb + w2*oof_lgbm + w3*oof_mlp
            if r_hit(pred, y_abs) > best_hit:
                best_hit, best_w = r_hit(pred, y_abs), (w1, w2, w3)
    print(f"  최적 가중치: XGB={best_w[0]:.2f}  LGBM={best_w[1]:.2f}  MLP={best_w[2]:.2f}", flush=True)
    print(f"  앙상블 R-Hit@1cm: {best_hit:.5f}", flush=True)
    return best_w


# ============================================================
# 메인 파이프라인
# ============================================================
def main():
    print(f"Device: {DEVICE}", flush=True)
    print("=" * 55, flush=True)

    # ── 1. 데이터 로드 ─────────────────────────────────────
    label_df  = pd.read_csv(LABEL_CSV)
    train_ids = label_df["id"].values
    y_abs     = label_df[["x", "y", "z"]].values.astype(np.float32)

    print("학습 데이터 로딩...", flush=True)
    X_traj = load_trajectories(TRAIN_DIR, train_ids, "Train")
    print(f"  train shape: {X_traj.shape}", flush=True)

    # ── 2. 정준 변환 ──────────────────────────────────────
    print("정준 좌표계 변환 중...", flush=True)
    X_can, R_all, centers = canonical_normalize_batch(X_traj)
    y_can = np.einsum('nij,nj->ni', R_all, (y_abs - centers))  # (N,3) canonical label

    # ── 3. 정준 프레임 다항식 예측 ─────────────────────────
    print(f"정준 다항식 예측 (deg={POLY_DEG}, t=+{FUTURE_MS:.0f}ms)...", flush=True)
    poly_can = poly_extrap_can_batch(X_can)
    # 원본 좌표로 복원하여 baseline 확인
    poly_orig = canonical_to_original(poly_can, R_all, centers)
    print(f"  Poly baseline R-Hit@1cm: {r_hit(poly_orig, y_abs):.5f}", flush=True)

    # ── 4. 정준 피처 추출 (XGB/LGBM용) ────────────────────
    print("피처 추출 중 (XGB/LGBM용)...", flush=True)
    X_feat = extract_all_flat(X_can, poly_can)  # (N, 196)
    y_res_can = (y_can - poly_can).astype(np.float32)  # canonical 잔차
    print(f"  피처 차원: {X_feat.shape}", flush=True)

    # ── 5. 5-Fold 학습 ────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_xgb  = np.zeros_like(y_abs)
    oof_lgbm = np.zeros_like(y_abs)
    oof_mlp  = np.zeros_like(y_abs)

    fold_xgbs, fold_lgbs, fold_mlps = [], [], []

    print(f"\n{N_FOLDS}-Fold 학습 시작...", flush=True)
    for fold, (tr_i, val_i) in enumerate(kf.split(X_feat)):
        print(f"\n{'='*40}\n[Fold {fold+1}/{N_FOLDS}]", flush=True)

        # XGBoost (canonical 잔차 예측)
        xgb_mdls, xgb_can_val = train_xgb_fold(
            X_feat[tr_i], y_res_can[tr_i],
            X_feat[val_i], y_res_can[val_i]
        )
        pred_can_xgb  = poly_can[val_i] + xgb_can_val
        oof_xgb[val_i] = canonical_to_original(pred_can_xgb, R_all[val_i], centers[val_i])
        fold_xgbs.append(xgb_mdls)
        print(f"  XGB  R-Hit={r_hit(oof_xgb[val_i], y_abs[val_i]):.5f}", flush=True)

        # LightGBM
        lgb_mdls, lgb_can_val = train_lgb_fold(
            X_feat[tr_i], y_res_can[tr_i],
            X_feat[val_i], y_res_can[val_i]
        )
        pred_can_lgb   = poly_can[val_i] + lgb_can_val
        oof_lgbm[val_i] = canonical_to_original(pred_can_lgb, R_all[val_i], centers[val_i])
        fold_lgbs.append(lgb_mdls)
        print(f"  LGBM R-Hit={r_hit(oof_lgbm[val_i], y_abs[val_i]):.5f}", flush=True)

        # MLP (증강 데이터)
        mlp, sc, mlp_val_orig, h_mlp = train_mlp_fold(
            X_can[tr_i], y_res_can[tr_i], poly_can[tr_i],
            X_can[val_i], poly_can[val_i], R_all[val_i], centers[val_i], y_abs[val_i]
        )
        oof_mlp[val_i] = mlp_val_orig
        fold_mlps.append((mlp, sc))
        print(f"  MLP  R-Hit={h_mlp:.5f}", flush=True)

    print(f"\n단순 평균 OOF: {r_hit((oof_xgb+oof_lgbm+oof_mlp)/3, y_abs):.5f}", flush=True)

    # ── 6. 가중치 최적화 ──────────────────────────────────
    print("\n[앙상블 가중치 최적화]", flush=True)
    w = optimize_weights(oof_xgb, oof_lgbm, oof_mlp, y_abs)
    final_oof = w[0]*oof_xgb + w[1]*oof_lgbm + w[2]*oof_mlp
    print(f"\n최종 OOF R-Hit@1cm: {r_hit(final_oof, y_abs):.5f}", flush=True)

    # ── 7. 테스트 추론 ────────────────────────────────────
    print("\n테스트 데이터 로딩...", flush=True)
    subm_df  = pd.read_csv(SUBM_CSV)
    test_ids = subm_df["id"].values
    X_test   = load_trajectories(TEST_DIR, test_ids, "Test")

    X_can_te, R_te, centers_te = canonical_normalize_batch(X_test)
    poly_can_te = poly_extrap_can_batch(X_can_te)
    Xf_te       = extract_all_flat(X_can_te, poly_can_te)

    # XGB 테스트
    xgb_can_preds = []
    for mdls in fold_xgbs:
        corr_can = np.stack([m.predict(Xf_te) for m in mdls], axis=-1)
        xgb_can_preds.append(canonical_to_original(
            poly_can_te + corr_can, R_te, centers_te))
    pred_xgb_te = np.mean(xgb_can_preds, axis=0)

    # LGBM 테스트
    lgb_can_preds = []
    for mdls in fold_lgbs:
        corr_can = np.stack([m.predict(Xf_te) for m in mdls], axis=-1)
        lgb_can_preds.append(canonical_to_original(
            poly_can_te + corr_can, R_te, centers_te))
    pred_lgbm_te = np.mean(lgb_can_preds, axis=0)

    # MLP 테스트
    mlp_preds_te = []
    for mlp, sc in fold_mlps:
        Xf_val_sc = sc.transform(extract_all_flat(X_can_te, poly_can_te)).astype(np.float32)
        mlp.eval()
        with torch.no_grad():
            corr_can = mlp(torch.from_numpy(Xf_val_sc).to(DEVICE)).cpu().numpy()
        mlp_preds_te.append(canonical_to_original(
            poly_can_te + corr_can, R_te, centers_te))
    pred_mlp_te = np.mean(mlp_preds_te, axis=0)

    final = w[0]*pred_xgb_te + w[1]*pred_lgbm_te + w[2]*pred_mlp_te

    # ── 8. 제출 파일 저장 ────────────────────────────────
    subm_df[["x", "y", "z"]] = final
    subm_df.to_csv(OUT_CSV, index=False)
    print(f"\n제출 파일 저장: {OUT_CSV}", flush=True)
    print(f"최종 OOF R-Hit@1cm: {r_hit(final_oof, y_abs):.5f}", flush=True)


if __name__ == "__main__":
    main()
