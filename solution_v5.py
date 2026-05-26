#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모기 비행 궤적 예측 v5 - 강화된 피처 + 대형 MLP + 소프트 R-Hit 손실
======================================================================

v4 대비 개선:
  1. 피처 확장 (196 → 277차원, 완전 벡터화 추출)
     - Snap (4차 미분), 속도/가속도/저크 크기 시계열
     - 구심/접선 가속도 분해, 각속도
     - 다중 다항식 예측 (deg=1, 2, 3)
  2. 대형 MLP: 1024-wide + skip connection
  3. 소프트 R-Hit 손실 (ep 180+에서 MSE와 50:50 혼합)
     - 거리 보존 (회전 불변) → 정준 프레임에서 직접 계산 가능
  4. 데이터 증강 ×15 (N_AUG=14)
  5. XGB/LGBM n_estimators=3000, 더 깊은 트리
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_DIR  = Path("./open")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
LABEL_CSV = DATA_DIR / "train_labels.csv"
SUBM_CSV  = DATA_DIR / "sample_submission.csv"
OUT_CSV   = Path("./submission.csv")

SEED      = 42
N_FOLDS   = 5
BATCH     = 1024
EPOCHS    = 300
LR        = 1e-3
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_AUG     = 14           # 1+14=15× 훈련 데이터

POLY_DEG  = 2
FUTURE_MS = 80.0
T_IN      = np.arange(-400, 1, 40, dtype=float)

SOFT_START = int(EPOCHS * 0.6)   # ep 180+에서 소프트 손실 혼합
SOFT_ALPHA = 0.5                  # MSE/soft 비율 50:50
SOFT_SIGMA = 0.005                # sigmoid 온도: 0.5cm

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
    N = len(trajs)
    centers = trajs[:, -1, :].copy()
    X_can   = np.zeros_like(trajs)
    R_all   = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        center = centers[i]
        cen    = trajs[i] - center
        vel    = cen[-1] - cen[-2]
        R      = _rotation_to_x(vel)
        X_can[i]  = cen @ R.T
        R_all[i]  = R
    return X_can.astype(np.float32), R_all, centers.astype(np.float32)


def canonical_to_original(pred_can: np.ndarray, R_all: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.einsum('nji,nj->ni', R_all, pred_can) + centers


# ============================================================
# 다항식 외삽 (정준 프레임, 다항식 예측 벡터 사전 계산)
# ============================================================
def _poly_pred_vec(times: np.ndarray, degree: int, t_future: float) -> np.ndarray:
    """times에서 degree 다항식 피팅 후 t_future 예측 계수 벡터 반환"""
    V = np.column_stack([times**d for d in range(degree, -1, -1)])
    Vinv = np.linalg.pinv(V)              # (degree+1, len(times))
    p = np.array([t_future**d for d in range(degree, -1, -1)])  # (degree+1,)
    return (p @ Vinv).astype(np.float64)  # (len(times),)


_PV1   = _poly_pred_vec(T_IN,      1, FUTURE_MS)   # (11,) deg=1
_PV2   = _poly_pred_vec(T_IN,      2, FUTURE_MS)   # (11,) deg=2
_PV3   = _poly_pred_vec(T_IN,      3, FUTURE_MS)   # (11,) deg=3
_PV2L5 = _poly_pred_vec(T_IN[-5:], 2, FUTURE_MS)   # (5,)  local deg=2
_PV2L9_T40 = _poly_pred_vec(T_IN[:-2], 2, -40.0)   # (9,)  LOO at t=-40
_PV2L9_T0  = _poly_pred_vec(T_IN[:-2], 2,   0.0)   # (9,)  LOO at t=0


def poly_extrap_can_batch(X_can: np.ndarray) -> np.ndarray:
    """(N,11,3) → (N,3) 정준 다항식 예측 (deg=2, t=+80ms)"""
    return (_PV2 @ X_can.astype(np.float64)).astype(np.float32)  # einsum over time axis


# ============================================================
# 피처 추출 (완전 벡터화, 277차원)
# ============================================================
def extract_all_flat(X_can: np.ndarray, poly_can: np.ndarray) -> np.ndarray:
    """
    (N,11,3) + (N,3) → (N,277) 피처 행렬
    피처 구성:
      cen(33) vel(30) acc(27) jerk(24) snap(21) direc(30) curv_cross(27) curv_mag(9)
      spd_series(10) acc_mag(9) jerk_mag(8) centripetal(9) tangential(9) ang_vel(9)
      poly_r(3) poly1(3) poly3(3) last_spd(1) local_lin(3) local_poly5(3) loo(6)
    """
    eps = 1e-8
    N = len(X_can)
    cen  = X_can.astype(np.float64)           # (N,11,3)

    vel  = np.diff(cen,  axis=1)              # (N,10,3)
    acc  = np.diff(vel,  axis=1)              # (N, 9,3)
    jerk = np.diff(acc,  axis=1)              # (N, 8,3)
    snap = np.diff(jerk, axis=1)              # (N, 7,3)

    spd   = np.linalg.norm(vel,  axis=-1, keepdims=True)  # (N,10,1)
    direc = vel / (spd + eps)                              # (N,10,3)
    curv_cross = np.cross(vel[:, :-1, :], acc)             # (N, 9,3)
    curv_mag   = np.linalg.norm(curv_cross, axis=-1)       # (N, 9)
    spd_series = spd[:, :, 0]                               # (N,10)
    acc_mag    = np.linalg.norm(acc,  axis=-1)              # (N, 9)
    jerk_mag   = np.linalg.norm(jerk, axis=-1)              # (N, 8)

    # 구심/접선 가속도 분해
    d_next     = direc[:, 1:, :]             # (N, 9,3) — acc 시점의 방향
    tang_proj  = np.sum(acc * d_next, axis=-1)               # (N, 9) 접선
    centripetal = np.linalg.norm(acc - tang_proj[:, :, None] * d_next, axis=-1)  # (N,9)

    # 각속도 (방향 변화율, rad/step)
    cos_ang = np.clip(np.sum(direc[:, :-1, :] * direc[:, 1:, :], axis=-1), -1.0, 1.0)  # (N,9)
    ang_vel = np.arccos(cos_ang)             # (N, 9)

    # 다항식 예측 (사전 계산 벡터 활용)
    poly_r     = np.einsum('t,nti->ni', _PV2,   cen).astype(np.float32)   # (N,3)
    poly1      = np.einsum('t,nti->ni', _PV1,   cen).astype(np.float32)   # (N,3)
    poly3      = np.einsum('t,nti->ni', _PV3,   cen).astype(np.float32)   # (N,3)
    local_poly5 = np.einsum('t,nti->ni', _PV2L5, cen[:, -5:, :]).astype(np.float32)  # (N,3)

    loo_t40 = np.einsum('t,nti->ni', _PV2L9_T40, cen[:, :-2, :])  # (N,3)
    loo_t0  = np.einsum('t,nti->ni', _PV2L9_T0,  cen[:, :-2, :])  # (N,3)
    loo_err_t40 = (loo_t40 - cen[:, -2, :]).astype(np.float32)  # (N,3)
    loo_err_t0  = (loo_t0  - cen[:, -1, :]).astype(np.float32)  # (N,3)
    loo = np.stack([loo_err_t40, loo_err_t0], axis=-1).reshape(N, 6)  # (N,6)

    local_lin = (2.0 * vel[:, -1, :]).astype(np.float32)   # (N,3)

    return np.concatenate([
        cen.reshape(N, -1).astype(np.float32),        # 33
        vel.reshape(N, -1).astype(np.float32),         # 30
        acc.reshape(N, -1).astype(np.float32),         # 27
        jerk.reshape(N, -1).astype(np.float32),        # 24
        snap.reshape(N, -1).astype(np.float32),        # 21
        direc.reshape(N, -1).astype(np.float32),       # 30
        curv_cross.reshape(N, -1).astype(np.float32),  # 27
        curv_mag.astype(np.float32),                   # 9
        spd_series.astype(np.float32),                 # 10
        acc_mag.astype(np.float32),                    # 9
        jerk_mag.astype(np.float32),                   # 8
        centripetal.astype(np.float32),                # 9
        tang_proj.astype(np.float32),                  # 9
        ang_vel.astype(np.float32),                    # 9
        poly_can.astype(np.float32),                   # 3  (poly_r = poly_can in canonical)
        poly1,                                         # 3
        poly3,                                         # 3
        spd_series[:, -1:].astype(np.float32),        # 1
        local_lin,                                     # 3
        local_poly5,                                   # 3
        loo,                                           # 6
    ], axis=1)  # (N, 277)


# ============================================================
# 데이터 증강 (x축 회전, MLP 전용)
# ============================================================
def augment_around_x(X_can: np.ndarray, y_can: np.ndarray, poly_can: np.ndarray,
                     n_aug: int = N_AUG) -> tuple:
    N = len(X_can)
    all_X, all_y, all_p = [X_can], [y_can], [poly_can]
    for _ in range(n_aug):
        th   = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
        c, s = np.cos(th), np.sin(th)
        Rx = np.zeros((N, 3, 3), dtype=np.float32)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = c;  Rx[:, 1, 2] = -s
        Rx[:, 2, 1] = s;  Rx[:, 2, 2] = c
        X_aug = np.einsum('nij,ntj->nti', Rx, X_can)
        y_aug = np.einsum('nij,nj->ni',   Rx, y_can)
        p_aug = np.einsum('nij,nj->ni',   Rx, poly_can)
        all_X.append(X_aug); all_y.append(y_aug); all_p.append(p_aug)
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
            n_estimators=3000, max_depth=7, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=100, random_state=SEED,
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
            n_estimators=3000, num_leaves=127, learning_rate=0.02,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr, y_can_tr[:, axis],
              eval_set=[(X_val, y_can_val[:, axis])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(period=0)])
        val_preds[:, axis] = m.predict(X_val)
        models.append(m)
    return models, val_preds


# ============================================================
# MLP (skip connection, soft R-Hit loss)
# ============================================================
class TrajDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


class BigResidualMLP(nn.Module):
    """1024-wide MLP with skip connection from input to fc3 output"""
    def __init__(self, in_dim: int = 277):
        super().__init__()
        self.fc1  = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.25))
        self.fc2  = nn.Sequential(nn.Linear(1024,   1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.20))
        self.fc3  = nn.Sequential(nn.Linear(1024,    512), nn.LayerNorm(512),  nn.GELU(), nn.Dropout(0.15))
        self.skip = nn.Linear(in_dim, 512)
        self.fc4  = nn.Sequential(nn.Linear(512,     256), nn.LayerNorm(256),  nn.GELU(), nn.Dropout(0.10))
        self.out  = nn.Linear(256, 3)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h) + self.skip(x)
        h = self.fc4(h)
        return self.out(h)


def soft_r_hit_loss(pred_res: torch.Tensor, true_res: torch.Tensor,
                    sigma: float = SOFT_SIGMA) -> torch.Tensor:
    """
    회전 불변성: 정준 잔차 거리 = 원본 거리
    → 1cm hit rate를 정준 프레임에서 직접 최대화
    """
    dist = torch.norm(pred_res - true_res, dim=-1)
    return -torch.sigmoid((0.01 - dist) / sigma).mean()


def train_mlp_fold(X_can_tr, y_can_tr, poly_can_tr,
                   X_can_val, poly_can_val, R_val, centers_val, y_abs_val):
    PATIENCE = 60

    X_aug, y_aug, p_aug = augment_around_x(X_can_tr, y_can_tr, poly_can_tr)
    print(f"    MLP 증강 훈련 샘플: {len(X_aug):,}", flush=True)

    Xf_tr  = extract_all_flat(X_aug,    p_aug)
    Xf_val = extract_all_flat(X_can_val, poly_can_val)

    sc   = StandardScaler().fit(Xf_tr)
    Xts  = sc.transform(Xf_tr).astype(np.float32)
    Xvs  = sc.transform(Xf_val).astype(np.float32)
    Xvt  = torch.from_numpy(Xvs).to(DEVICE)

    y_aug_t   = y_aug.astype(np.float32)
    dl        = DataLoader(TrajDataset(Xts, y_aug_t),
                           batch_size=BATCH, shuffle=True, num_workers=0)
    in_dim    = Xts.shape[1]
    model     = BigResidualMLP(in_dim=in_dim).to(DEVICE)
    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # poly_can_val as tensor for soft-loss validation
    pvcv_t    = torch.from_numpy(poly_can_val.astype(np.float32)).to(DEVICE)

    best_hit, best_state, no_imp = -1.0, None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            mse  = F.mse_loss(pred, yb)
            if ep >= SOFT_START:
                loss = (1 - SOFT_ALPHA) * mse + SOFT_ALPHA * soft_r_hit_loss(pred, yb)
            else:
                loss = mse
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                can_corr = model(Xvt).cpu().numpy()
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

    label_df  = pd.read_csv(LABEL_CSV)
    train_ids = label_df["id"].values
    y_abs     = label_df[["x", "y", "z"]].values.astype(np.float32)

    print("학습 데이터 로딩...", flush=True)
    X_traj = load_trajectories(TRAIN_DIR, train_ids, "Train")
    print(f"  train shape: {X_traj.shape}", flush=True)

    print("정준 좌표계 변환 중...", flush=True)
    X_can, R_all, centers = canonical_normalize_batch(X_traj)
    y_can = np.einsum('nij,nj->ni', R_all, (y_abs - centers))

    print(f"정준 다항식 예측 (deg={POLY_DEG}, t=+{FUTURE_MS:.0f}ms)...", flush=True)
    poly_can  = poly_extrap_can_batch(X_can)
    poly_orig = canonical_to_original(poly_can, R_all, centers)
    print(f"  Poly baseline R-Hit@1cm: {r_hit(poly_orig, y_abs):.5f}", flush=True)

    print("피처 추출 중 (XGB/LGBM용)...", flush=True)
    X_feat    = extract_all_flat(X_can, poly_can)
    y_res_can = (y_can - poly_can).astype(np.float32)
    print(f"  피처 차원: {X_feat.shape}", flush=True)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_xgb  = np.zeros_like(y_abs)
    oof_lgbm = np.zeros_like(y_abs)
    oof_mlp  = np.zeros_like(y_abs)
    fold_xgbs, fold_lgbs, fold_mlps = [], [], []

    print(f"\n{N_FOLDS}-Fold 학습 시작...", flush=True)
    for fold, (tr_i, val_i) in enumerate(kf.split(X_feat)):
        print(f"\n{'='*40}\n[Fold {fold+1}/{N_FOLDS}]", flush=True)

        xgb_mdls, xgb_can_val = train_xgb_fold(
            X_feat[tr_i], y_res_can[tr_i],
            X_feat[val_i], y_res_can[val_i])
        pred_can_xgb   = poly_can[val_i] + xgb_can_val
        oof_xgb[val_i] = canonical_to_original(pred_can_xgb, R_all[val_i], centers[val_i])
        fold_xgbs.append(xgb_mdls)
        print(f"  XGB  R-Hit={r_hit(oof_xgb[val_i], y_abs[val_i]):.5f}", flush=True)

        lgb_mdls, lgb_can_val = train_lgb_fold(
            X_feat[tr_i], y_res_can[tr_i],
            X_feat[val_i], y_res_can[val_i])
        pred_can_lgb    = poly_can[val_i] + lgb_can_val
        oof_lgbm[val_i] = canonical_to_original(pred_can_lgb, R_all[val_i], centers[val_i])
        fold_lgbs.append(lgb_mdls)
        print(f"  LGBM R-Hit={r_hit(oof_lgbm[val_i], y_abs[val_i]):.5f}", flush=True)

        mlp, sc, mlp_val_orig, h_mlp = train_mlp_fold(
            X_can[tr_i], y_res_can[tr_i], poly_can[tr_i],
            X_can[val_i], poly_can[val_i], R_all[val_i], centers[val_i], y_abs[val_i])
        oof_mlp[val_i] = mlp_val_orig
        fold_mlps.append((mlp, sc))
        print(f"  MLP  R-Hit={h_mlp:.5f}", flush=True)

    print(f"\n단순 평균 OOF: {r_hit((oof_xgb+oof_lgbm+oof_mlp)/3, y_abs):.5f}", flush=True)

    print("\n[앙상블 가중치 최적화]", flush=True)
    w = optimize_weights(oof_xgb, oof_lgbm, oof_mlp, y_abs)
    final_oof = w[0]*oof_xgb + w[1]*oof_lgbm + w[2]*oof_mlp
    print(f"\n최종 OOF R-Hit@1cm: {r_hit(final_oof, y_abs):.5f}", flush=True)

    print("\n테스트 데이터 로딩...", flush=True)
    subm_df  = pd.read_csv(SUBM_CSV)
    test_ids = subm_df["id"].values
    X_test   = load_trajectories(TEST_DIR, test_ids, "Test")

    X_can_te, R_te, centers_te = canonical_normalize_batch(X_test)
    poly_can_te = poly_extrap_can_batch(X_can_te)
    Xf_te       = extract_all_flat(X_can_te, poly_can_te)

    xgb_can_preds = []
    for mdls in fold_xgbs:
        corr_can = np.stack([m.predict(Xf_te) for m in mdls], axis=-1)
        xgb_can_preds.append(canonical_to_original(
            poly_can_te + corr_can, R_te, centers_te))
    pred_xgb_te = np.mean(xgb_can_preds, axis=0)

    lgb_can_preds = []
    for mdls in fold_lgbs:
        corr_can = np.stack([m.predict(Xf_te) for m in mdls], axis=-1)
        lgb_can_preds.append(canonical_to_original(
            poly_can_te + corr_can, R_te, centers_te))
    pred_lgbm_te = np.mean(lgb_can_preds, axis=0)

    mlp_preds_te = []
    for mlp, sc in fold_mlps:
        Xf_sc = sc.transform(extract_all_flat(X_can_te, poly_can_te)).astype(np.float32)
        mlp.eval()
        with torch.no_grad():
            corr_can = mlp(torch.from_numpy(Xf_sc).to(DEVICE)).cpu().numpy()
        mlp_preds_te.append(canonical_to_original(
            poly_can_te + corr_can, R_te, centers_te))
    pred_mlp_te = np.mean(mlp_preds_te, axis=0)

    final = w[0]*pred_xgb_te + w[1]*pred_lgbm_te + w[2]*pred_mlp_te

    subm_df[["x", "y", "z"]] = final
    subm_df.to_csv(OUT_CSV, index=False)
    print(f"\n제출 파일 저장: {OUT_CSV}", flush=True)
    print(f"최종 OOF R-Hit@1cm: {r_hit(final_oof, y_abs):.5f}", flush=True)


if __name__ == "__main__":
    main()
