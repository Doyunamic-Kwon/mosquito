#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모기 비행 궤적 예측 - R-Hit@1cm 최적화 솔루션
================================================

평가 지표: R-Hit@1cm
  - 예측 좌표와 실제 좌표의 3D 유클리드 거리 d <= 0.01m 이면 적중(1)
  - 전체 샘플 평균 적중률(0~1)

핵심 전략:
  1. 다항식 외삽(Polynomial Extrapolation) — 물리 기반 기저 예측
  2. MLP 잔차 보정(Residual Correction)    — 비선형 오차 추가 학습
  3. 5-Fold 앙상블                         — 일반화 성능 향상

데이터:
  - 입력: 11 타임스텝 (t=-400ms ~ 0ms, 40ms 간격) x (x,y,z) 좌표
  - 출력: 미래 시점의 (x,y,z) 좌표 (최적 시점은 학습 데이터에서 자동 탐색)

개발 환경:
  OS: macOS 14.x / Ubuntu 22.04
  Python: 3.10.x
  torch: 2.8.0
  numpy: 2.0.2
  pandas: 2.3.3
  scikit-learn: 1.6.1
  tqdm: 4.67.3
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

# ============================================================
# 하이퍼파라미터
# ============================================================
SEED    = 42
N_FOLDS = 5
BATCH   = 512
EPOCHS  = 300
LR      = 1e-3
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# 입력 시계열: -400ms ~ 0ms, 40ms 간격 (11개 포인트)
T_IN    = np.arange(-400, 1, 40, dtype=float)
N_STEPS = len(T_IN)  # 11

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 평가 지표
# ============================================================
def r_hit(pred: np.ndarray, true: np.ndarray, thr: float = 0.01) -> float:
    """3D 유클리드 거리 <= thr 인 비율 반환"""
    dist = np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)
    return float(np.mean(dist <= thr))


# ============================================================
# 데이터 로딩
# ============================================================
def load_trajectories(directory: Path, ids: np.ndarray, desc: str = "") -> np.ndarray:
    """CSV 파일 목록 → (N, 11, 3) float32 배열"""
    seqs = []
    for id_ in tqdm(ids, desc=desc, ncols=80):
        df = pd.read_csv(directory / f"{id_}.csv")
        seqs.append(df[["x", "y", "z"]].values.astype(np.float32))
    return np.stack(seqs)


# ============================================================
# 다항식 외삽 (물리 기반 기저 예측)
# ============================================================
def poly_extrap_one(traj: np.ndarray, deg: int, t_future: float) -> np.ndarray:
    """traj (11,3) 에 각 축별 deg차 다항식 피팅 → t_future에서 외삽 → (3,)"""
    pred = np.empty(3, dtype=np.float32)
    for i in range(3):
        coef = np.polyfit(T_IN, traj[:, i], deg)
        pred[i] = float(np.polyval(coef, t_future))
    return pred


def poly_extrap_batch(trajs: np.ndarray, deg: int, t_future: float) -> np.ndarray:
    """(N,11,3) → (N,3)"""
    return np.stack([poly_extrap_one(t, deg, t_future) for t in trajs])


def search_best_poly(trajs: np.ndarray, labels: np.ndarray) -> tuple:
    """
    학습 데이터에서 최적 (다항식 차수, 예측 시점) 자동 탐색.
    반환: (best_deg, best_future_ms)
    """
    print("\n[다항식 파라미터 탐색]")
    best_hit, best_deg, best_ms = -1.0, 3, 80.0
    for deg in [1, 2, 3, 4]:
        for ms in [40, 80, 120, 160, 200, 240]:
            preds = poly_extrap_batch(trajs, deg, float(ms))
            hit = r_hit(preds, labels)
            mark = " ★" if hit > best_hit else ""
            print(f"  deg={deg}, t=+{ms:3d}ms  R-Hit={hit:.5f}{mark}")
            if hit > best_hit:
                best_hit, best_deg, best_ms = hit, deg, float(ms)
    print(f"\n  최적: deg={best_deg}, t=+{best_ms:.0f}ms → R-Hit={best_hit:.5f}")
    return best_deg, best_ms


# ============================================================
# 피처 추출
# ============================================================
def build_features(traj: np.ndarray, poly_pred: np.ndarray) -> np.ndarray:
    """
    traj (11,3), poly_pred (3,) → 94차원 특징 벡터

    구성:
      [  0- 32] t=0 기준 중심화 좌표 (11×3 = 33)
      [ 33- 62] 속도 (10×3 = 30)
      [ 63- 89] 가속도 ( 9×3 = 27)
      [ 90- 92] 다항식 예측 잔차 = poly_pred - last (3)
      [    93 ] t=0 순간속도 크기 (1)
    """
    last     = traj[-1]
    centered = (traj - last).astype(np.float32)
    vel      = np.diff(centered, axis=0)            # (10, 3)
    acc      = np.diff(vel,      axis=0)            # ( 9, 3)
    poly_r   = (poly_pred - last).astype(np.float32)
    speed    = np.array([np.linalg.norm(vel[-1])], dtype=np.float32)
    return np.concatenate([centered.ravel(), vel.ravel(), acc.ravel(), poly_r, speed])


def extract_all(trajs: np.ndarray, poly_preds: np.ndarray) -> np.ndarray:
    """(N,11,3), (N,3) → (N,94) float32"""
    return np.stack([build_features(trajs[i], poly_preds[i]) for i in range(len(trajs))]).astype(np.float32)


# ============================================================
# 신경망 (MLP 잔차 보정기)
# ============================================================
class TrajDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


class ResidualMLP(nn.Module):
    """다항식 예측의 잔차(label - poly_pred)를 학습하는 MLP"""
    def __init__(self, in_dim: int = 94):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, 512),    nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128),                       nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 학습 루프 (단일 Fold)
# ============================================================
def train_fold(
    X_tr: np.ndarray, y_res_tr: np.ndarray,
    X_val: np.ndarray, poly_val: np.ndarray, y_abs_val: np.ndarray,
) -> tuple:
    """
    잔차 타겟(y_res = label - poly_pred)으로 모델 학습.
    반환: (model, scaler, best_val_r_hit)
    """
    scaler  = StandardScaler().fit(X_tr)
    Xts     = scaler.transform(X_tr).astype(np.float32)
    Xvs     = scaler.transform(X_val).astype(np.float32)

    dl_tr = DataLoader(
        TrajDataset(Xts, y_res_tr.astype(np.float32)),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )

    model   = ResidualMLP(in_dim=Xts.shape[1]).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    best_hit, best_state, no_imp = -1.0, None, 0
    Xvt = torch.from_numpy(Xvs).to(DEVICE)

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # 10에포크마다 R-Hit 평가
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                corr = model(Xvt).cpu().numpy()
            hit = r_hit(poly_val + corr, y_abs_val)
            if hit > best_hit:
                best_hit  = hit
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 10
            if no_imp >= 50:  # patience=50 에포크
                print(f"    조기종료 ep={ep:3d} | best={best_hit:.5f}")
                break

        if ep % 100 == 0:
            print(f"    ep={ep:3d} | best R-Hit={best_hit:.5f}")

    model.load_state_dict(best_state)
    return model, scaler, best_hit


# ============================================================
# 메인 파이프라인
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    print("=" * 55)

    # ── 1. 데이터 로드 ─────────────────────────────────────
    label_df  = pd.read_csv(LABEL_CSV)
    train_ids = label_df["id"].values
    y_abs     = label_df[["x", "y", "z"]].values.astype(np.float32)

    print("학습 궤적 로딩 중...")
    X_traj = load_trajectories(TRAIN_DIR, train_ids, "Train")
    print(f"  train shape: {X_traj.shape}")  # (10000, 11, 3)

    # ── 2. 최적 다항식 파라미터 탐색 ──────────────────────
    poly_deg, future_ms = search_best_poly(X_traj, y_abs)

    # ── 3. 다항식 기저 예측 ────────────────────────────────
    print(f"\n다항식 예측 계산 (deg={poly_deg}, t=+{future_ms:.0f}ms)...")
    poly_tr = poly_extrap_batch(X_traj, poly_deg, future_ms)
    print(f"  Poly baseline R-Hit@1cm: {r_hit(poly_tr, y_abs):.5f}")

    # ── 4. 피처 추출 ──────────────────────────────────────
    print("피처 추출 중...")
    X_feat = extract_all(X_traj, poly_tr)       # (10000, 94)
    y_res  = (y_abs - poly_tr).astype(np.float32)  # 잔차 타겟
    print(f"  feature shape: {X_feat.shape}")

    # ── 5. 5-Fold 학습 ────────────────────────────────────
    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof     = np.zeros_like(y_abs)
    models  = []
    scalers = []

    print(f"\n{N_FOLDS}-Fold 학습 시작...")
    for fold, (tr_i, val_i) in enumerate(kf.split(X_feat)):
        print(f"\n[Fold {fold + 1}/{N_FOLDS}]")
        mdl, scl, hit = train_fold(
            X_feat[tr_i], y_res[tr_i],
            X_feat[val_i], poly_tr[val_i], y_abs[val_i],
        )
        # OOF 예측 저장
        mdl.eval()
        with torch.no_grad():
            Xvt = torch.from_numpy(scl.transform(X_feat[val_i]).astype(np.float32)).to(DEVICE)
            oof[val_i] = poly_tr[val_i] + mdl(Xvt).cpu().numpy()
        models.append(mdl)
        scalers.append(scl)
        print(f"  Fold {fold + 1} Best R-Hit@1cm: {hit:.5f}")

    oof_hit = r_hit(oof, y_abs)
    print(f"\n{'=' * 55}")
    print(f"OOF R-Hit@1cm: {oof_hit:.5f}")
    print(f"{'=' * 55}")

    # ── 6. 테스트 추론 ────────────────────────────────────
    print("\n테스트 궤적 로딩 중...")
    subm_df  = pd.read_csv(SUBM_CSV)
    test_ids = subm_df["id"].values
    X_test   = load_trajectories(TEST_DIR, test_ids, "Test")
    poly_te  = poly_extrap_batch(X_test, poly_deg, future_ms)
    X_test_f = extract_all(X_test, poly_te)

    preds_list = []
    for mdl, scl in zip(models, scalers):
        mdl.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(scl.transform(X_test_f).astype(np.float32)).to(DEVICE)
            preds_list.append(poly_te + mdl(Xt).cpu().numpy())

    # 5-fold 평균 앙상블
    final = np.mean(preds_list, axis=0)

    # ── 7. 제출 파일 저장 ────────────────────────────────
    subm_df[["x", "y", "z"]] = final
    subm_df.to_csv(OUT_CSV, index=False)
    print(f"\n제출 파일 저장 완료: {OUT_CSV}")
    print(f"(OOF R-Hit@1cm: {oof_hit:.5f})")


if __name__ == "__main__":
    main()
