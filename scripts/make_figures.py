"""진단 그림 생성: 측정 프로파일, 존 영향함수, CV 예측 품질."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import nested_cv, rmse
from mrrpred.pipeline import MRRPredictor

d = load("data/Train_1.csv")
r, Y, P = d.radius, d.mrr, d.pressure
o = np.argsort(r)
FIG = pathlib.Path("figures"); FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. 원데이터
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
lo = P[:, 0] < 2.6
for ax, mask, ttl in [(axes[0], lo, "Low-pressure group (wafer 1-15)"),
                      (axes[1], ~lo, "High-pressure group (wafer 16-21)")]:
    for i in np.where(mask)[0]:
        ax.plot(r[o], Y[i][o], lw=1.1,
                label=f"#{d.wafer_id[i]} ({P[i,0]:g},{P[i,1]:g},{P[i,2]:g})")
    ax.set_xlabel("Radius [mm]"); ax.set_ylabel("MRR [A/min]")
    ax.set_title(ttl); ax.grid(alpha=.3)
    ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "01_measured_profiles.png", dpi=130)

# --------------------------------------------- 2. 존 영향함수 dMRR/dP_zone(r)
m = M.RadialRidge(alpha=1.0, feat="zone+ring").fit(P, Y, radius=r)
B = m.B_[1:] / m.xs_[:, None]          # 표준화 해제 -> 물리 단위 [A/min per psi]
fig, ax = plt.subplots(figsize=(9, 5))
for k, (nm, c) in enumerate(zip(["Zone 1 (edge)", "Zone 2 (mid)", "Zone 3 (center)",
                                 "Retainer ring"], ["C3", "C1", "C0", "C7"])):
    ax.plot(r[o], B[k][o], lw=2, color=c, label=nm)
ax.axhline(0, c="k", lw=.7)
for b in (55, 68):
    ax.axvline(b, ls=":", c="gray", lw=.8); ax.axvline(-b, ls=":", c="gray", lw=.8)
ax.set_xlabel("Radius [mm]"); ax.set_ylabel(r"$\partial$MRR/$\partial P$  [A/min per psi]")
ax.set_title("Zone influence functions (learned from data)")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "02_influence_functions.png", dpi=130)

# ------------------------------------------------------- 3. CV 예측 vs 실측
best = M.SymmetryRidge
space = dict(alpha=[0.01, .1, 1, 3, 10, 30, 100], feat=["zone", "zone+ring", "quad", "quad+ring", "phys"],
             rank=[0, 1, 2, 3], alpha_a=[0.1, 1, 10], smooth=[0.0])
pred, chosen = nested_cv(best, space, P, Y, r, d.design_groups(), inner_groups=d.design_groups())
print("LOGO-CV RMSE =", round(rmse(Y, pred), 1))

sel = [0, 6, 8, 9, 12, 15, 17, 20]
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for ax, i in zip(axes.ravel(), sel):
    ax.plot(r[o], Y[i][o], "k-o", ms=2.5, lw=1.2, label="measured")
    ax.plot(r[o], pred[i][o], "r--", lw=1.6, label="predicted (unseen)")
    ax.set_title(f"#{d.wafer_id[i]}  P=({P[i,0]:g},{P[i,1]:g},{P[i,2]:g})  "
                 f"RMSE={rmse(Y[i],pred[i]):.0f}", fontsize=9)
    ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Radius [mm]")
for ax in axes[:, 0]:
    ax.set_ylabel("MRR [A/min]")
fig.suptitle("Leave-one-design-out cross-validation (each wafer predicted without its own design)")
fig.tight_layout(); fig.savefig(FIG / "03_cv_profiles.png", dpi=130)

# ------------------------------------------- 4. 반경별 오차 vs 반복실험 노이즈
g = d.design_groups()
pe = []
for gi in np.unique(g):
    mk = g == gi
    if mk.sum() > 1:
        pe.append(((Y[mk] - Y[mk].mean(0)) ** 2).sum(0) / (mk.sum() - 1))
pe = np.sqrt(np.mean(pe, 0))
err = np.sqrt(((Y - pred) ** 2).mean(0))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(r[o], err[o], "-o", ms=3, label="model CV error")
ax.plot(r[o], pe[o], "-s", ms=3, label="repeat-experiment noise (floor)")
ax.set_xlabel("Radius [mm]"); ax.set_ylabel("RMSE [A/min]")
ax.set_title("Where the error lives: model vs irreducible measurement noise")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "04_error_vs_noise.png", dpi=130)

print("figures/ 에 4개 그림 저장 완료")
