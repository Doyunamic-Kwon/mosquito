#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모기 비행 궤적 예측 v6 - 순수 딥러닝 (Transformer + MLP)
==========================================================

아키텍처: TrajPredictor
  - Transformer Branch: 정준 궤적 (B,11,3) → (B,256)
  - Feature Branch:     277차원 피처      → (B,512)
  - Fusion Head:        (B,768) → (B,3) [정준 잔차]

핵심:
  1. 정준 좌표계 (Rodrigues 회전, 방향 불변)
  2. ×50 데이터 증강 (x축 회전)
  3. 소프트 R-Hit 손실 (ep200+ MSE와 혼합)
  4. AMP (GPU) + 5-Fold 앙상블

GPU 서버 실행 권장. CPU 가능하나 느림.
"""

import sys, warnings
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

warnings.filterwarnings("ignore")

DATA_DIR  = Path("./open")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
LABEL_CSV = DATA_DIR / "train_labels.csv"
SUBM_CSV  = DATA_DIR / "sample_submission.csv"
OUT_CSV   = Path("./submission.csv")

SEED      = 42
N_FOLDS   = 5
N_AUG     = 49          # ×50 augmentation
EPOCHS    = 500
BATCH     = 2048        # GPU 권장; CPU라면 512로 줄이기
LR        = 3e-4
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP   = (DEVICE == "cuda")

D_MODEL   = 256
NHEAD     = 8
N_LAYERS  = 4
DIM_FF    = 1024

SOFT_START = 200        # ep 200부터 소프트 R-Hit 혼합
SOFT_ALPHA = 0.5
SOFT_SIGMA = 0.005
PATIENCE   = 80

POLY_DEG  = 2
FUTURE_MS = 80.0
T_IN      = np.arange(-400, 1, 40, dtype=float)  # 11 timesteps

np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Device: {DEVICE}  AMP: {USE_AMP}", flush=True)


# ── 평가 지표 ────────────────────────────────────────────────────────────
def r_hit(pred, true, thr=0.01):
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= thr))


# ── 데이터 로딩 ──────────────────────────────────────────────────────────
def load_trajectories(directory, ids, desc=""):
    seqs = []
    for id_ in tqdm(ids, desc=desc, ncols=80):
        df = pd.read_csv(directory / f"{id_}.csv")
        seqs.append(df[["x", "y", "z"]].values.astype(np.float32))
    return np.stack(seqs)


# ── 정준 좌표계 ──────────────────────────────────────────────────────────
def _rotation_to_x(vel):
    v_norm = np.linalg.norm(vel)
    if v_norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    v = vel / v_norm
    cos_t = float(v[0])
    cross = np.cross(v, [1., 0., 0.])
    sin_t = np.linalg.norm(cross)
    if sin_t < 1e-8:
        return np.eye(3, dtype=np.float32) if cos_t > 0 else np.diag([-1., 1., -1.]).astype(np.float32)
    k = cross / sin_t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return (np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)).astype(np.float32)


def canonical_normalize_batch(trajs):
    N = len(trajs)
    centers = trajs[:, -1, :].copy()
    X_can   = np.zeros_like(trajs)
    R_all   = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        cen      = trajs[i] - centers[i]
        R        = _rotation_to_x(cen[-1] - cen[-2])
        X_can[i] = cen @ R.T
        R_all[i] = R
    return X_can.astype(np.float32), R_all, centers.astype(np.float32)


def canonical_to_original(pred_can, R_all, centers):
    return np.einsum('nji,nj->ni', R_all, pred_can) + centers


# ── 다항식 예측 벡터 (사전계산) ──────────────────────────────────────────
def _poly_pred_vec(times, degree, t_future):
    V = np.column_stack([times**d for d in range(degree, -1, -1)])
    p = np.array([t_future**d for d in range(degree, -1, -1)])
    return (p @ np.linalg.pinv(V)).astype(np.float64)

_PV1       = _poly_pred_vec(T_IN,       1, FUTURE_MS)
_PV2       = _poly_pred_vec(T_IN,       2, FUTURE_MS)
_PV3       = _poly_pred_vec(T_IN,       3, FUTURE_MS)
_PV2L5     = _poly_pred_vec(T_IN[-5:],  2, FUTURE_MS)
_PV2L9_T40 = _poly_pred_vec(T_IN[:-2],  2, -40.0)
_PV2L9_T0  = _poly_pred_vec(T_IN[:-2],  2,   0.0)


def poly_extrap_batch(X_can):
    """(N,11,3) → (N,3)  deg=2 예측"""
    return np.einsum('t,nti->ni', _PV2, X_can.astype(np.float64)).astype(np.float32)


# ── 피처 추출 (277차원, 완전 벡터화) ────────────────────────────────────
def extract_features(X_can):
    """(N,11,3) → (N,277)  poly_can은 내부에서 재계산"""
    eps = 1e-8
    N   = len(X_can)
    cen = X_can.astype(np.float64)

    vel  = np.diff(cen,  axis=1)             # (N,10,3)
    acc  = np.diff(vel,  axis=1)             # (N, 9,3)
    jerk = np.diff(acc,  axis=1)             # (N, 8,3)
    snap = np.diff(jerk, axis=1)             # (N, 7,3)

    spd        = np.linalg.norm(vel, axis=-1, keepdims=True)   # (N,10,1)
    direc      = vel / (spd + eps)
    curv_cross = np.cross(vel[:, :-1], acc)
    curv_mag   = np.linalg.norm(curv_cross, axis=-1)
    acc_mag    = np.linalg.norm(acc,  axis=-1)
    jerk_mag   = np.linalg.norm(jerk, axis=-1)

    d_next  = direc[:, 1:]
    tang    = np.sum(acc * d_next, axis=-1)
    cent    = np.linalg.norm(acc - tang[:, :, None] * d_next, axis=-1)
    cos_ang = np.clip(np.sum(direc[:, :-1] * direc[:, 1:], axis=-1), -1.0, 1.0)
    ang_vel = np.arccos(cos_ang)

    poly2  = np.einsum('t,nti->ni', _PV2,      cen)
    poly1  = np.einsum('t,nti->ni', _PV1,      cen)
    poly3  = np.einsum('t,nti->ni', _PV3,      cen)
    loc5   = np.einsum('t,nti->ni', _PV2L5,    cen[:, -5:])
    lt40   = np.einsum('t,nti->ni', _PV2L9_T40, cen[:, :-2])
    lt0    = np.einsum('t,nti->ni', _PV2L9_T0,  cen[:, :-2])
    loo    = np.stack([(lt40 - cen[:, -2]).astype(np.float32),
                       (lt0  - cen[:, -1]).astype(np.float32)], axis=-1).reshape(N, 6)

    return np.concatenate([
        cen.reshape(N, -1).astype(np.float32),         # 33
        vel.reshape(N, -1).astype(np.float32),          # 30
        acc.reshape(N, -1).astype(np.float32),          # 27
        jerk.reshape(N, -1).astype(np.float32),         # 24
        snap.reshape(N, -1).astype(np.float32),         # 21
        direc.reshape(N, -1).astype(np.float32),        # 30
        curv_cross.reshape(N, -1).astype(np.float32),   # 27
        curv_mag.astype(np.float32),                    # 9
        spd[:, :, 0].astype(np.float32),                # 10
        acc_mag.astype(np.float32),                     # 9
        jerk_mag.astype(np.float32),                    # 8
        cent.astype(np.float32),                        # 9
        tang.astype(np.float32),                        # 9
        ang_vel.astype(np.float32),                     # 9
        poly2.astype(np.float32),                       # 3
        poly1.astype(np.float32),                       # 3
        poly3.astype(np.float32),                       # 3
        spd[:, -1, :].astype(np.float32),               # 1  (last speed, keepdims=1)
        (2.0 * vel[:, -1]).astype(np.float32),          # 3
        loc5.astype(np.float32),                        # 3
        loo,                                            # 6
    ], axis=1)   # (N, 277)


# ── 데이터 증강 ──────────────────────────────────────────────────────────
def augment_x_rot(X_can, y_res_can, n_aug=N_AUG):
    """x축 회전으로 (1+n_aug)× 증강. y_res_can = y_can - poly_can."""
    N = len(X_can)
    Xs, Ys = [X_can], [y_res_can]
    for _ in range(n_aug):
        th = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
        c, s = np.cos(th), np.sin(th)
        Rx = np.zeros((N, 3, 3), dtype=np.float32)
        Rx[:, 0, 0] = 1.
        Rx[:, 1, 1] = c;  Rx[:, 1, 2] = -s
        Rx[:, 2, 1] = s;  Rx[:, 2, 2] = c
        Xs.append(np.einsum('nij,ntj->nti', Rx, X_can))
        Ys.append(np.einsum('nij,nj->ni',   Rx, y_res_can))
    return np.concatenate(Xs, 0), np.concatenate(Ys, 0)


# ── 모델 ─────────────────────────────────────────────────────────────────
class TransformerBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj    = nn.Linear(3, D_MODEL)
        self.pos_emb = nn.Embedding(11, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=0.1, norm_first=True, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYERS)

    def forward(self, x):                        # (B,11,3)
        pos = torch.arange(11, device=x.device)
        h   = self.proj(x) + self.pos_emb(pos)
        return self.enc(h).mean(dim=1)           # (B, D_MODEL)


class FeatureBranch(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.fc1  = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2))
        self.fc2  = nn.Sequential(nn.Linear(1024,   1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.2))
        self.fc3  = nn.Sequential(nn.Linear(1024, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc3(h) + self.skip(x)       # (B, out_dim)


class TrajPredictor(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.t_branch = TransformerBranch()
        self.f_branch = FeatureBranch(feat_dim)
        fused = D_MODEL + 512
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512,   256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256,     3),
        )

    def forward(self, traj, feat):
        return self.head(torch.cat([self.t_branch(traj), self.f_branch(feat)], dim=-1))


# ── 손실 ─────────────────────────────────────────────────────────────────
def soft_r_hit(pred_res, true_res, sigma=SOFT_SIGMA):
    """거리 보존(회전 불변) → 정준 잔차 거리 = 원본 거리"""
    return -torch.sigmoid((0.01 - torch.norm(pred_res - true_res, dim=-1)) / sigma).mean()


# ── 데이터셋 ─────────────────────────────────────────────────────────────
class TrajDataset(Dataset):
    def __init__(self, X_can, X_feat, y=None):
        self.tc = torch.from_numpy(X_can.astype(np.float32))
        self.tf = torch.from_numpy(X_feat.astype(np.float32))
        self.ty = torch.from_numpy(y.astype(np.float32)) if y is not None else None

    def __len__(self): return len(self.tc)

    def __getitem__(self, i):
        if self.ty is not None:
            return self.tc[i], self.tf[i], self.ty[i]
        return self.tc[i], self.tf[i]


# ── 폴드 학습 ────────────────────────────────────────────────────────────
def train_fold(X_can_tr, y_res_tr, X_can_val, poly_val, R_val, centers_val, y_abs_val):
    # 증강
    X_aug, y_aug = augment_x_rot(X_can_tr, y_res_tr)
    print(f"    증강 샘플: {len(X_aug):,}", flush=True)

    # 피처 추출 (augmented poly는 내부 재계산)
    poly_aug = poly_extrap_batch(X_aug)
    y_aug_res = y_aug  # y_aug = Rx @ y_res → augmented residual
    # BUT augmented poly != Rx @ poly_tr, so recalculate the target properly:
    # y_can_aug = Rx @ y_can = Rx @ (y_res + poly_tr)
    # poly_aug  = Rx @ poly_tr
    # correct y_aug_res = y_can_aug - poly_aug = Rx@y_res ✓  (already what augment_x_rot returns)

    Xf_tr  = extract_features(X_aug)
    Xf_val = extract_features(X_can_val)

    sc     = StandardScaler().fit(Xf_tr)
    Xfs_tr = sc.transform(Xf_tr).astype(np.float32)
    Xfs_vl = sc.transform(Xf_val).astype(np.float32)

    nw  = 4 if DEVICE == "cuda" else 0
    pm  = (DEVICE == "cuda")
    dl  = DataLoader(TrajDataset(X_aug, Xfs_tr, y_aug_res),
                     batch_size=BATCH, shuffle=True, num_workers=nw, pin_memory=pm)

    Xc_vt = torch.from_numpy(X_can_val.astype(np.float32)).to(DEVICE)
    Xf_vt = torch.from_numpy(Xfs_vl).to(DEVICE)

    model  = TrajPredictor(feat_dim=Xfs_tr.shape[1]).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_hit, best_state, no_imp = -1.0, None, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for tb, fb, yb in dl:
            tb = tb.to(DEVICE, non_blocking=True)
            fb = fb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred = model(tb, fb)
                loss = F.mse_loss(pred, yb)
                if ep >= SOFT_START:
                    loss = (1 - SOFT_ALPHA) * loss + SOFT_ALPHA * soft_r_hit(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    corr = model(Xc_vt, Xf_vt).float().cpu().numpy()
            pred_orig = canonical_to_original(poly_val + corr, R_val, centers_val)
            hit = r_hit(pred_orig, y_abs_val)
            if hit > best_hit:
                best_hit, no_imp = hit, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 10
            if no_imp >= PATIENCE:
                print(f"    조기종료 ep={ep:3d}  best={best_hit:.5f}", flush=True)
                break
        if ep % 100 == 0:
            print(f"    ep={ep:3d}  best={best_hit:.5f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        corr = model(Xc_vt, Xf_vt).cpu().numpy()
    return model, sc, canonical_to_original(poly_val + corr, R_val, centers_val), best_hit


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55, flush=True)

    label_df  = pd.read_csv(LABEL_CSV)
    train_ids = label_df["id"].values
    y_abs     = label_df[["x", "y", "z"]].values.astype(np.float32)

    print("학습 데이터 로딩...", flush=True)
    X_traj = load_trajectories(TRAIN_DIR, train_ids, "Train")

    print("정준 변환...", flush=True)
    X_can, R_all, centers = canonical_normalize_batch(X_traj)
    y_can     = np.einsum('nij,nj->ni', R_all, (y_abs - centers))
    poly_can  = poly_extrap_batch(X_can)
    poly_orig = canonical_to_original(poly_can, R_all, centers)
    print(f"  Poly baseline: {r_hit(poly_orig, y_abs):.5f}", flush=True)

    y_res_can = (y_can - poly_can).astype(np.float32)

    kf   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof  = np.zeros_like(y_abs)
    mdls = []

    print(f"\n{N_FOLDS}-Fold 학습 (Transformer+MLP, ×{N_AUG+1} aug)...", flush=True)
    for fold, (tr_i, val_i) in enumerate(kf.split(X_can)):
        print(f"\n{'='*40}\n[Fold {fold+1}/{N_FOLDS}]", flush=True)
        model, sc, val_pred, h = train_fold(
            X_can[tr_i], y_res_can[tr_i],
            X_can[val_i], poly_can[val_i], R_all[val_i], centers[val_i], y_abs[val_i])
        oof[val_i] = val_pred
        mdls.append((model, sc))
        print(f"  Fold {fold+1}  R-Hit={h:.5f}", flush=True)

    print(f"\n최종 OOF R-Hit@1cm: {r_hit(oof, y_abs):.5f}", flush=True)

    print("\n테스트 추론...", flush=True)
    subm_df  = pd.read_csv(SUBM_CSV)
    X_test   = load_trajectories(TEST_DIR, subm_df["id"].values, "Test")
    X_can_te, R_te, ctr_te = canonical_normalize_batch(X_test)
    poly_te  = poly_extrap_batch(X_can_te)
    Xc_te_t  = torch.from_numpy(X_can_te).to(DEVICE)

    preds = []
    for model, sc in mdls:
        Xf_te = sc.transform(extract_features(X_can_te)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            corr = model(Xc_te_t, torch.from_numpy(Xf_te).to(DEVICE)).cpu().numpy()
        preds.append(canonical_to_original(poly_te + corr, R_te, ctr_te))

    final = np.mean(preds, axis=0)
    subm_df[["x", "y", "z"]] = final
    subm_df.to_csv(OUT_CSV, index=False)
    print(f"제출 파일 저장: {OUT_CSV}", flush=True)
    print(f"최종 OOF R-Hit@1cm: {r_hit(oof, y_abs):.5f}", flush=True)


if __name__ == "__main__":
    main()
