#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모기 비행 궤적 예측 v2 - R-Hit@1cm 0.70+ 목표
================================================

v1 대비 개선 사항:
  1. 멀티스케일 피처 (184 → 196차원)
     - 곡률 벡터·크기, 방향 벡터, 저크(3차 미분)
     - 로컬 선형 예측 (최근 2포인트), 로컬 2차 다항식 (최근 5포인트)
     - LOO 적합 오차 (처음 9포인트로 피팅, 마지막 2포인트에서 오차)
  2. 모델 앙상블: MLP + BiLSTM(128) + 1D-CNN (fold당 3개, 총 30개 예측 평균)
  3. 10-Fold CV (5→10) — 앙상블 다양성 극대화
  4. 소프트 R-Hit 손실: MSE + sigmoid 근사 (후반 20%에서 혼합)
  5. 에포크 400, patience 60

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
SEED      = 42
N_FOLDS   = 10          # 10-fold: 앙상블 다양성 ↑
BATCH     = 512
EPOCHS    = 400
PATIENCE  = 60          # 조기종료 patience (10 에포크 배수)
LR        = 1e-3
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# 소프트 R-Hit 손실 혼합 시작 비율 (후반 20%부터)
SOFT_START = int(EPOCHS * 0.8)

# 다항식 외삽 (v1 탐색 결과)
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
    return np.stack(seqs)   # (N, 11, 3)


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
# 피처 추출 (MLP용 — 196차원)
# ============================================================
def build_features(traj: np.ndarray, poly_pred: np.ndarray) -> np.ndarray:
    """
    196차원 피처 벡터:
      [  0- 32] centered coord (11×3=33)      — t=0 기준 중심화 좌표
      [ 33- 62] velocity       (10×3=30)      — 1차 차분
      [ 63- 89] acceleration   ( 9×3=27)      — 2차 차분
      [ 90-113] jerk           ( 8×3=24)      — 3차 차분
      [114-143] direction      (10×3=30)      — 단위 속도 벡터
      [144-170] curv_cross     ( 9×3=27)      — vel[:-1] × acc
      [171-179] curv_mag       (     9)       — |curv_cross|
      [180-182] poly_residual  (     3)       — global poly - last
      [    183] speed          (     1)       — t=0 순간속도 크기
      [184-186] local_linear   (     3)       — 최근 2스텝 선형 예측 (centered)
      [187-189] local_poly5    (     3)       — 최근 5포인트 2차 예측 (centered)
      [190-195] loo_errors     (     6)       — LOO 적합 오차 (t=-40, t=0, 3축)
    """
    eps   = 1e-8
    last  = traj[-1]
    cen   = (traj - last).astype(np.float32)            # (11, 3)
    vel   = np.diff(cen, axis=0)                        # (10, 3)
    acc   = np.diff(vel, axis=0)                        # ( 9, 3)
    jerk  = np.diff(acc, axis=0)                        # ( 8, 3)
    spd   = np.linalg.norm(vel, axis=-1, keepdims=True) # (10, 1)
    direc = vel / (spd + eps)                           # (10, 3)
    curv_cross = np.cross(vel[:-1], acc)                # ( 9, 3)
    curv_mag   = np.linalg.norm(curv_cross, axis=-1)    # ( 9,)
    poly_r     = (poly_pred - last).astype(np.float32)

    # 로컬 선형 예측: 최근 속도 × 2스텝 (centered, t=+80ms)
    local_lin = (2.0 * vel[-1]).astype(np.float32)      # (3,)

    # 로컬 2차 다항식: 최근 5포인트 피팅, t=+80ms (centered 기준)
    local_poly5 = np.empty(3, dtype=np.float32)
    for i in range(3):
        coef5 = np.polyfit(T_IN[-5:], cen[-5:, i], 2)
        local_poly5[i] = float(np.polyval(coef5, FUTURE_MS))

    # LOO 적합 오차: 처음 9포인트로 피팅 → t=-40, t=0에서 오차 (절대좌표)
    loo = np.empty(6, dtype=np.float32)
    for i in range(3):
        coef_loo = np.polyfit(T_IN[:-2], traj[:-2, i], POLY_DEG)
        loo[i * 2]     = float(np.polyval(coef_loo, -40)) - traj[-2, i]
        loo[i * 2 + 1] = float(np.polyval(coef_loo,   0)) - traj[-1, i]

    return np.concatenate([
        cen.ravel(), vel.ravel(), acc.ravel(), jerk.ravel(),
        direc.ravel(), curv_cross.ravel(), curv_mag,
        poly_r, [spd[-1, 0]],
        local_lin, local_poly5, loo,
    ]).astype(np.float32)   # 196


def extract_all_flat(trajs: np.ndarray, poly_preds: np.ndarray) -> np.ndarray:
    return np.stack([build_features(trajs[i], poly_preds[i])
                     for i in range(len(trajs))]).astype(np.float32)


# ============================================================
# 시퀀스 입력 준비 (LSTM · CNN용)
# ============================================================
def build_seq_input(trajs: np.ndarray) -> np.ndarray:
    """(N, 11, 3) 중심화 궤적 반환"""
    return np.stack([trajs[i] - trajs[i, -1]
                     for i in range(len(trajs))]).astype(np.float32)


class SeqScaler:
    """(N,11,3) → (N,11,3) 정규화 (flatten → StandardScaler → reshape)"""
    def __init__(self):
        self.sc = StandardScaler()

    def fit(self, X: np.ndarray):
        self.sc.fit(X.reshape(len(X), -1))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        N, T, F = X.shape
        return self.sc.transform(X.reshape(N, -1)).reshape(N, T, F).astype(np.float32)


# ============================================================
# 손실 함수
# ============================================================
def compute_loss(pred: torch.Tensor, target: torch.Tensor, epoch: int) -> torch.Tensor:
    """
    초반 80%: 순수 MSE
    후반 20%: 0.5×MSE + 0.5×소프트 R-Hit (temperature=0.004)
    """
    dist = torch.norm(pred - target, dim=-1)
    mse  = (dist ** 2).mean()
    if epoch < SOFT_START:
        return mse
    T        = 0.004
    sr_loss  = 1.0 - torch.sigmoid((0.01 - dist) / T).mean()
    return 0.5 * mse + 0.5 * sr_loss


# ============================================================
# 모델 정의
# ============================================================
class ResidualMLP(nn.Module):
    """확장 피처(196차원) MLP"""
    def __init__(self, in_dim: int = 196):
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


class BiLSTM(nn.Module):
    """Bidirectional LSTM: 중심화 궤적 (N,11,3) → 잔차 (N,3)"""
    def __init__(self, input_size: int = 3, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True, dropout=0.2, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class Conv1DCNN(nn.Module):
    """1D-CNN: (N,3,11) → 잔차 (N,3)"""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(3,   64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,  128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x.permute(0, 2, 1)))   # (N,11,3)→(N,3,11)


# ============================================================
# Dataset
# ============================================================
class TrajDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


# ============================================================
# 단일 모델 학습
# ============================================================
def train_one(model: nn.Module,
              X_tr: np.ndarray, y_res_tr: np.ndarray,
              X_val_t: torch.Tensor,
              poly_val: np.ndarray, y_abs_val: np.ndarray,
              tag: str = "") -> float:
    """반환: best_val_r_hit"""
    dl = DataLoader(
        TrajDataset(X_tr, y_res_tr.astype(np.float32)),
        batch_size=BATCH, shuffle=True, num_workers=0,
    )
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_hit, best_state, no_imp = -1.0, None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            compute_loss(model(xb), yb, ep).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                corr = model(X_val_t).cpu().numpy()
            hit = r_hit(poly_val + corr, y_abs_val)
            if hit > best_hit:
                best_hit  = hit
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 10
            if no_imp >= PATIENCE:
                print(f"    [{tag}] 조기종료 ep={ep:3d} best={best_hit:.5f}")
                break

        if ep % 100 == 0:
            print(f"    [{tag}] ep={ep:3d} | best={best_hit:.5f}")

    model.load_state_dict(best_state)
    return best_hit


# ============================================================
# Fold 학습 (MLP + LSTM + CNN)
# ============================================================
def train_fold_all(
    X_flat_tr, X_seq_tr, y_res_tr,
    X_flat_val, X_seq_val,
    poly_val, y_abs_val,
):
    """반환: (oof_pred, mlp_info, lstm_info, cnn_info)"""

    # ── MLP ──────────────────────────────────────────────────
    mlp_sc = StandardScaler().fit(X_flat_tr)
    Xfts   = mlp_sc.transform(X_flat_tr).astype(np.float32)
    Xfvs   = mlp_sc.transform(X_flat_val).astype(np.float32)
    Xfv_t  = torch.from_numpy(Xfvs).to(DEVICE)
    mlp    = ResidualMLP(in_dim=Xfts.shape[1]).to(DEVICE)
    h_mlp  = train_one(mlp, Xfts, y_res_tr, Xfv_t, poly_val, y_abs_val, "MLP")
    print(f"  MLP  R-Hit={h_mlp:.5f}")

    # ── BiLSTM ───────────────────────────────────────────────
    lstm_sc = SeqScaler().fit(X_seq_tr)
    Xsts    = lstm_sc.transform(X_seq_tr)
    Xsvs    = lstm_sc.transform(X_seq_val)
    Xsv_t   = torch.from_numpy(Xsvs).to(DEVICE)
    lstm    = BiLSTM().to(DEVICE)
    h_lstm  = train_one(lstm, Xsts, y_res_tr, Xsv_t, poly_val, y_abs_val, "LSTM")
    print(f"  LSTM R-Hit={h_lstm:.5f}")

    # ── 1D-CNN ───────────────────────────────────────────────
    cnn_sc = SeqScaler().fit(X_seq_tr)
    Xcts   = cnn_sc.transform(X_seq_tr)
    Xcvs   = cnn_sc.transform(X_seq_val)
    Xcv_t  = torch.from_numpy(Xcvs).to(DEVICE)
    cnn    = Conv1DCNN().to(DEVICE)
    h_cnn  = train_one(cnn, Xcts, y_res_tr, Xcv_t, poly_val, y_abs_val, "CNN")
    print(f"  CNN  R-Hit={h_cnn:.5f}")

    # ── OOF 앙상블 (3모델 평균) ──────────────────────────
    mlp.eval(); lstm.eval(); cnn.eval()
    with torch.no_grad():
        c_mlp  = mlp(Xfv_t).cpu().numpy()
        c_lstm = lstm(Xsv_t).cpu().numpy()
        c_cnn  = cnn(Xcv_t).cpu().numpy()
    oof = poly_val + (c_mlp + c_lstm + c_cnn) / 3.0
    print(f"  앙상블 R-Hit={r_hit(oof, y_abs_val):.5f}")

    return oof, (mlp, mlp_sc), (lstm, lstm_sc), (cnn, cnn_sc)


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
    print(f"  train shape: {X_traj.shape}")

    # ── 2. 다항식 기저 예측 ────────────────────────────────
    print(f"다항식 예측 (deg={POLY_DEG}, t=+{FUTURE_MS:.0f}ms)...")
    poly_tr = poly_extrap_batch(X_traj)
    print(f"  Poly baseline R-Hit@1cm: {r_hit(poly_tr, y_abs):.5f}")

    # ── 3. 피처 준비 ──────────────────────────────────────
    print("피처 추출 중...")
    X_flat = extract_all_flat(X_traj, poly_tr)       # (N, 196)
    X_seq  = build_seq_input(X_traj)                 # (N, 11, 3)
    y_res  = (y_abs - poly_tr).astype(np.float32)
    print(f"  MLP feature dim: {X_flat.shape[1]}, Seq: {X_seq.shape}")

    # ── 4. K-Fold 학습 ────────────────────────────────────
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof        = np.zeros_like(y_abs)
    fold_artes = []

    print(f"\n{N_FOLDS}-Fold 학습 시작...")
    for fold, (tr_i, val_i) in enumerate(kf.split(X_flat)):
        print(f"\n{'='*40}\n[Fold {fold+1}/{N_FOLDS}]")
        oof_fold, mlp_i, lstm_i, cnn_i = train_fold_all(
            X_flat[tr_i], X_seq[tr_i], y_res[tr_i],
            X_flat[val_i], X_seq[val_i],
            poly_tr[val_i], y_abs[val_i],
        )
        oof[val_i] = oof_fold
        fold_artes.append((mlp_i, lstm_i, cnn_i))

    oof_hit = r_hit(oof, y_abs)
    print(f"\n{'='*55}")
    print(f"최종 OOF R-Hit@1cm: {oof_hit:.5f}")
    print(f"{'='*55}")

    # ── 5. 테스트 추론 ────────────────────────────────────
    print("\n테스트 궤적 로딩 중...")
    subm_df  = pd.read_csv(SUBM_CSV)
    test_ids = subm_df["id"].values
    X_test   = load_trajectories(TEST_DIR, test_ids, "Test")
    poly_te  = poly_extrap_batch(X_test)
    Xf_te    = extract_all_flat(X_test, poly_te)
    Xs_te    = build_seq_input(X_test)

    all_preds = []
    for mlp_i, lstm_i, cnn_i in fold_artes:
        mlp,  mlp_sc  = mlp_i
        lstm, lstm_sc = lstm_i
        cnn,  cnn_sc  = cnn_i
        mlp.eval(); lstm.eval(); cnn.eval()
        with torch.no_grad():
            Xft = torch.from_numpy(mlp_sc.transform(Xf_te).astype(np.float32)).to(DEVICE)
            Xst = torch.from_numpy(lstm_sc.transform(Xs_te)).to(DEVICE)
            Xct = torch.from_numpy(cnn_sc.transform(Xs_te)).to(DEVICE)
            c_mlp  = mlp(Xft).cpu().numpy()
            c_lstm = lstm(Xst).cpu().numpy()
            c_cnn  = cnn(Xct).cpu().numpy()
        all_preds.append(poly_te + (c_mlp + c_lstm + c_cnn) / 3.0)

    # 10 fold × 3 model = 30개 예측 평균
    final = np.mean(all_preds, axis=0)

    # ── 6. 제출 파일 저장 ────────────────────────────────
    subm_df[["x", "y", "z"]] = final
    subm_df.to_csv(OUT_CSV, index=False)
    print(f"\n제출 파일 저장: {OUT_CSV}")
    print(f"OOF R-Hit@1cm: {oof_hit:.5f}")


if __name__ == "__main__":
    main()
