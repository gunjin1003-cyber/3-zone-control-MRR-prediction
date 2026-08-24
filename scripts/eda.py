"""데이터 구조 진단: 존-반경 대응, 반복오차(노이즈 하한), PCA 차원."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load

d = load("data/Train_1.csv")
R, P, r = d.mrr, d.pressure, d.radius
np.set_printoptions(precision=1, suppress=True, linewidth=200)

print("=" * 78)
print("1) 반복 실험(동일 압력) 기반 순수 오차 = 어떤 모델도 못 넘는 하한")
g = d.design_groups()
resid, per_rad = [], []
for gi in np.unique(g):
    m = g == gi
    if m.sum() < 2:
        continue
    blk = R[m]
    print(f"   design {P[m][0,:3]}  n={m.sum()}  wafers={d.wafer_id[m]}")
    resid.append(blk - blk.mean(0))
    per_rad.append(((blk - blk.mean(0)) ** 2).sum(0) / (m.sum() - 1))
resid = np.vstack(resid)
dof = sum((g == gi).sum() - 1 for gi in np.unique(g) if (g == gi).sum() > 1)
pure = np.sqrt((resid ** 2).sum() / (dof * d.n_radius))
per_rad = np.sqrt(np.mean(per_rad, axis=0))
print(f"   -> pure-error RMSE = {pure:.1f} A/min  (전체 표준편차 {R.std():.1f})")
print(f"   -> 중심부(|r|<=50) {per_rad[np.abs(r) <= 50].mean():.0f} / "
      f"에지(|r|>=70) {per_rad[np.abs(r) >= 70].mean():.0f} A/min")

print("=" * 78)
print("2) 존별 압력 <-> 반경별 MRR 상관 (어느 존이 어느 반경을 지배하나)")
print("     r(mm) |  Zone1   Zone2   Zone3   Ring")
for i in range(0, d.n_radius, 3):
    c = [np.corrcoef(P[:, k], R[:, i])[0, 1] for k in range(4)]
    print(f"   {r[i]:6.0f} | " + "".join(f"{v:7.2f} " for v in c))

print("=" * 78)
print("3) 프로파일 PCA (선형 부분공간 차원)")
mu = R.mean(0)
U, s, Vt = np.linalg.svd(R - mu, full_matrices=False)
ev = s ** 2 / (s ** 2).sum()
print("   설명분산비:", np.round(ev[:8] * 100, 2))
print("   누적      :", np.round(np.cumsum(ev[:8]) * 100, 2))
resid_k = [np.sqrt(np.mean((s[k:] ** 2)) / d.n_radius * 0 + ((s[k:] ** 2).sum() / (d.n_wafer * d.n_radius))) for k in range(1, 7)]
print("   k개 성분 사용 시 재구성 RMSE:", [f"{v:.0f}" for v in resid_k])

print("=" * 78)
print("4) 레벨(평균 MRR) vs 압력 - Preston 선형성")
lvl = R.mean(1)
for k, n in enumerate(d.feature_names):
    print(f"   corr(mean MRR, {n:12s}) = {np.corrcoef(P[:, k], lvl)[0,1]: .3f}")
A = np.c_[np.ones(21), P[:, :3]]
coef, *_ = np.linalg.lstsq(A, lvl, rcond=None)
pred = A @ coef
print(f"   선형회귀 계수 b0={coef[0]:.0f}, Z1={coef[1]:.0f}, Z2={coef[2]:.0f}, Z3={coef[3]:.0f}")
print(f"   R2={1 - ((lvl-pred)**2).sum()/((lvl-lvl.mean())**2).sum():.4f}")

print("=" * 78)
print("5) 설계 공간 - 외삽 위험")
print("   저압군(웨이퍼 1-15) 압력범위:", P[:15, :3].min(0), "~", P[:15, :3].max(0))
print("   고압군(웨이퍼 16-21) 압력범위:", P[15:, :3].min(0), "~", P[15:, :3].max(0))
print("   평균 MRR: 저압군", lvl[:15].mean().round(0), "고압군", lvl[15:].mean().round(0))
