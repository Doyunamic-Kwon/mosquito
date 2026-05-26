#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모기 비행 궤적 예측 v3 - XGBoost + LightGBM + MLP 하이브리드
==============================================================

전략:
  1. 피처 196차원 (곡률, 방향, LOO 오차, 로컬 다항식 포함)
  2. XGBoost (축별 3개 모델) — 표 형식 데이터 강자
  3. LightGBM (축별 3개 모델) — XGB와 다른 편향 보정
  4. MLP (3축 공동 예측) — 비선형 상호작용 학습
  5. OOF 기반 앙상블 가중치 최적화
  6. 5-Fold CV, 예상 실행 시간 ~15분

개발 환경:
  Python: 3.10.x  torch: 2.8.0  numpy: 2.0.2
  pandas: 2.3.3  scikit-learn: 1.6.1
  xgboost: 설치 버전  lightgbm: 설치 버전  tqdm: 4.67.3
"""

import sys
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
BATCH   = 512
EPOCHS  = 300
LR      = 1e-3
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

POLY_DEG  = 2
FUTURE_MS = 80.0
T_IN      = np.arange(-400, 1, 40, dtype=float)   # 11개

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
# 다항식 외삽
# ============================================================
def poly_extrap_one(traj: np.ndarray, deg: int, t_future: float) -> np.ndarray:
    pred = np.empty(3, dtype=np.float32)
    for i in range(3):
        coef = np.polyfit(T_IN, traj[:, i], deg)
        pred[i] = float(np.polyval(coef, t_future))
    return pred


def poly_extrap_batch(trajs: np.ndarray, deg: int = POLY_DEG,
                      t_future: float = FUTURE_MS) -> np.ndarray:
    return np.stack([poly_extrap_one(t, deg, t_future) for t in trajs])


# ============================================================
# 피처 추출 (196차원)
# ============================================================
def build_features(traj: np.ndarray, poly_pred: np.ndarray) -> np.ndarray:
    """
    196차원:
      [  0- 32] centered coord (33)   [180-182] poly_residual (3)
      [ 33- 62] velocity (30)         [    183] speed (1)
      [ 63- 89] acceleration (27)     [184-186] local_linear (3)
      [ 90-113] jerk (24)             [187-189] local_poly5 (3)
      [114-143] direction (30)        [190-195] loo_errors (6)
      [144-170] curv_cross (27)
      [171-179] curv_mag (9)
    """
    eps   = 1e-8
    last  = traj[-1]
    cen   = (traj - last).astype(np.float32)
    vel   = np.diff(cen, axis=0)                         # (10,3)
    acc   = np.diff(vel, axis=0)                         # ( 9,3)
    jerk  = np.diff(acc, axis=0)                         # ( 8,3)
    spd   = np.linalg.norm(vel, axis=-1, keepdims=True)  # (10,1)
    direc = vel / (spd + eps)                            # (10,3)
    curv_cross = np.cross(vel[:-1], acc)                 # ( 9,3)
    curv_mag   = np.linalg.norm(curv_cross, axis=-1)     # ( 9,)
    poly_r     = (poly_pred - last).astype(np.float32)

    local_lin = (2.0 * vel[-1]).astype(np.float32)

    local_poly5 = np.empty(3, dtype=np.float32)
    for i in range(3):
        c = np.polyfit(T_IN[-5:], cen[-5:, i], 2)
        local_poly5[i] = float(np.polyval(c, FUTURE_MS))

    loo = np.empty(6, dtype=np.float32)
    for i in range(3):
        c = np.polyfit(T_IN[:-2], traj[:-2, i], POLY_DEG)
        loo[i * 2]     = float(np.polyval(c, -40)) - traj[-2, i]
        loo[i * 2 + 1] = float(np.polyval(c,   0)) - traj[-1, i]

    return np.concatenate([
        cen.ravel(), vel.ravel(), acc.ravel(), jerk.ravel(),
        direc.ravel(), curv_cross.ravel(), curv_mag,
        poly_r, [spd[-1, 0]],
        local_lin, local_poly5, loo,
    ]).astype(np.float32)


def extract_all_flat(trajs: np.ndarray, poly_preds: np.ndarray) -> np.ndarray:
    return np.stack([build_features(trajs[i], poly_preds[i])
                     for i in range(len(trajs))]).astype(np.float32)


# ============================================================
# XGBoost (축별 3개 모델)
# ============================================================
def train_xgb_fold(X_tr, y_res_tr, X_val, y_res_val):
    models, val_preds = [], np.zeros((len(X_val), 3))
    for axis in range(3):
        m = xgb.XGBRegressor(
            n_estimators=2000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            early_stopping_rounds=50, random_state=SEED,
            n_jobs=-1, verbosity=0,
        )
        m.fit(X_tr, y_res_tr[:, axis],
              eval_set=[(X_val, y_res_val[:, axis])],
              verbose=False)
        val_preds[:, axis] = m.predict(X_val)
        models.append(m)
    return models, val_preds


# ============================================================
# LightGBM (축별 3개 모델)
# ============================================================
def train_lgb_fold(X_tr, y_res_tr, X_val, y_res_val):
    models, val_preds = [], np.zeros((len(X_val), 3))
    for axis in range(3):
        m = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=63, learning_rate=0.03,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr, y_res_tr[:, axis],
              eval_set=[(X_val, y_res_val[:, axis])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=0)])
        val_preds[:, axis] = m.predict(X_val)
        models.append(m)
    return models, val_preds


# ============================================================
# MLP (3축 공동 예측)
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


def train_mlp_fold(X_tr, y_res_tr, X_val, poly_val, y_abs_val):
    PATIENCE = 40
    sc  = StandardScaler().fit(X_tr)
    Xts = sc.transform(X_tr).astype(np.float32)
    Xvs = sc.transform(X_val).astype(np.float32)
    Xvt = torch.from_numpy(Xvs).to(DEVICE)

    dl    = DataLoader(TrajDataset(Xts, y_res_tr.astype(np.float32)),
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
                corr = model(Xvt).cpu().numpy()
            hit = r_hit(poly_val + corr, y_abs_val)
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
        val_preds = poly_val + model(Xvt).cpu().numpy()
    return model, sc, val_preds, best_hit


# ============================================================
# 앙상블 가중치 최적화 (OOF 기반 그리드 탐색)
# ============================================================
def optimize_weights(oof_xgb, oof_lgbm, oof_mlp, y_abs):
    best_hit, best_w = -1.0, (1/3, 1/3, 1/3)
    for w1 in np.arange(0, 1.01, 0.1):
        for w2 in np.arange(0, 1.01 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 10)
            if w3 < -1e-9: continue
            w3 = max(0.0, w3)
            pred = w1 * oof_xgb + w2 * oof_lgbm + w3 * oof_mlp
            hit  = r_hit(pred, y_abs)
            if hit > best_hit:
                best_hit, best_w = hit, (w1, w2, w3)
    print(f"  최적 가중치: XGB={best_w[0]:.2f}  LGBM={best_w[1]:.2f}  MLP={best_w[2]:.2f}")
    print(f"  가중 앙상블 R-Hit@1cm: {best_hit:.5f}")
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

    # ── 2. 다항식 기저 예측 ────────────────────────────────
    print(f"다항식 예측 (deg={POLY_DEG}, t=+{FUTURE_MS:.0f}ms)...", flush=True)
    poly_tr = poly_extrap_batch(X_traj)
    print(f"  Poly baseline R-Hit@1cm: {r_hit(poly_tr, y_abs):.5f}", flush=True)

    # ── 3. 피처 추출 ──────────────────────────────────────
    print("피처 추출 중...", flush=True)
    X_feat = extract_all_flat(X_traj, poly_tr)
    y_res  = (y_abs - poly_tr).astype(np.float32)
    print(f"  피처 차원: {X_feat.shape}", flush=True)

    # ── 4. 5-Fold 학습 ────────────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_xgb  = np.zeros_like(y_abs)
    oof_lgbm = np.zeros_like(y_abs)
    oof_mlp  = np.zeros_like(y_abs)

    fold_xgbs, fold_lgbs, fold_mlps = [], [], []

    print(f"\n{N_FOLDS}-Fold 학습 시작...", flush=True)
    for fold, (tr_i, val_i) in enumerate(kf.split(X_feat)):
        print(f"\n{'='*40}\n[Fold {fold+1}/{N_FOLDS}]", flush=True)
        y_res_tr  = y_res[tr_i]
        y_res_val = y_res[val_i]

        # XGBoost
        xgb_mdls, xgb_val = train_xgb_fold(X_feat[tr_i], y_res_tr,
                                             X_feat[val_i], y_res_val)
        oof_xgb[val_i] = poly_tr[val_i] + xgb_val
        fold_xgbs.append(xgb_mdls)
        print(f"  XGB  R-Hit={r_hit(oof_xgb[val_i], y_abs[val_i]):.5f}", flush=True)

        # LightGBM
        lgb_mdls, lgb_val = train_lgb_fold(X_feat[tr_i], y_res_tr,
                                             X_feat[val_i], y_res_val)
        oof_lgbm[val_i] = poly_tr[val_i] + lgb_val
        fold_lgbs.append(lgb_mdls)
        print(f"  LGBM R-Hit={r_hit(oof_lgbm[val_i], y_abs[val_i]):.5f}", flush=True)

        # MLP
        mlp, sc, mlp_val, h_mlp = train_mlp_fold(
            X_feat[tr_i], y_res_tr, X_feat[val_i], poly_tr[val_i], y_abs[val_i]
        )
        oof_mlp[val_i] = mlp_val
        fold_mlps.append((mlp, sc))
        print(f"  MLP  R-Hit={h_mlp:.5f}", flush=True)

    print(f"\n단순 평균 OOF: {r_hit((oof_xgb+oof_lgbm+oof_mlp)/3, y_abs):.5f}", flush=True)

    # ── 5. 가중치 최적화 ──────────────────────────────────
    print("\n[앙상블 가중치 최적화]", flush=True)
    w = optimize_weights(oof_xgb, oof_lgbm, oof_mlp, y_abs)
    final_oof = w[0]*oof_xgb + w[1]*oof_lgbm + w[2]*oof_mlp
    print(f"\n최종 OOF R-Hit@1cm: {r_hit(final_oof, y_abs):.5f}", flush=True)

    # ── 6. 테스트 추론 ────────────────────────────────────
    print("\n테스트 데이터 로딩...", flush=True)
    subm_df  = pd.read_csv(SUBM_CSV)
    test_ids = subm_df["id"].values
    X_test   = load_trajectories(TEST_DIR, test_ids, "Test")
    poly_te  = poly_extrap_batch(X_test)
    Xf_te    = extract_all_flat(X_test, poly_te)

    # XGB 테스트 예측
    xgb_te = np.mean([
        poly_te + np.stack([m.predict(Xf_te) for m in mdls], axis=-1)
        for mdls in fold_xgbs
    ], axis=0)

    # LGBM 테스트 예측
    lgb_te = np.mean([
        poly_te + np.stack([m.predict(Xf_te) for m in mdls], axis=-1)
        for mdls in fold_lgbs
    ], axis=0)

    # MLP 테스트 예측
    mlp_te_list = []
    for mlp, sc in fold_mlps:
        mlp.eval()
        Xt = torch.from_numpy(sc.transform(Xf_te).astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            mlp_te_list.append(poly_te + mlp(Xt).cpu().numpy())
    mlp_te = np.mean(mlp_te_list, axis=0)

    final = w[0]*xgb_te + w[1]*lgb_te + w[2]*mlp_te

    # ── 7. 제출 파일 저장 ────────────────────────────────
    subm_df[["x", "y", "z"]] = final
    subm_df.to_csv(OUT_CSV, index=False)
    print(f"\n제출 파일 저장: {OUT_CSV}", flush=True)
    print(f"최종 OOF R-Hit@1cm: {r_hit(final_oof, y_abs):.5f}", flush=True)


if __name__ == "__main__":
    main()
