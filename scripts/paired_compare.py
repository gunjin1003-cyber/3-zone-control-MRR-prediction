"""상위 모델 쌍대(paired) 비교.

두 모델이 같은 웨이퍼를 예측하므로 오차가 강하게 상관된다. 독립 표준오차를
겹쳐보는 방식(±24)은 이 상관을 무시해 실제로 존재하는 작은 차이를 놓친다.
설계조건 fold 단위로 MSE 차이를 짝지어 부트스트랩하면 훨씬 민감하다.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import folds, rmse

d = load("data/Train_1.csv")
P, Y, r, g = d.pressure, d.mrr, d.radius, d.design_groups()


def cv(cls, prm):
    pred = np.zeros_like(Y)
    for tr, te in folds(g):
        m = cls(**prm)
        try:
            m.fit(P[tr], Y[tr], radius=r)
        except TypeError:
            m.fit(P[tr], Y[tr])
        pred[te] = m.predict(P[te])
    return pred


print("Quad Ridge 는 alpha 에 민감하다 — 먼저 제대로 잡는다")
for al in (0.1, 0.3, 1, 2, 3, 10):
    print(f"   alpha={al:<5g} RMSE {rmse(Y, cv(M.RadialRidge, dict(alpha=al, feat='quad'))):6.1f}")

BASE = ("Quad Ridge (a=1)", M.RadialRidge, dict(alpha=1.0, feat="quad"))
OTHERS = [
    ("대칭성 능형 [quad]", M.SymmetryRidge, dict(alpha=1.0, feat="quad", rank=1, alpha_a=1.0)),
    ("PLS 3성분 [quad]",   M.SklearnMulti,  dict(kind="pls", feat="quad", n_comp=3)),
    ("PCA(4)+Quad Ridge", M.PCARegressor,  dict(k=4, head="ridge", alpha=1.0, feat="quad")),
    ("Zone Ridge (1차)",   M.RadialRidge,   dict(alpha=1.0, feat="zone")),
]

gs = np.unique(g)
base = cv(BASE[1], BASE[2])
bm = np.array([((Y[g == gi] - base[g == gi]) ** 2).mean() for gi in gs])
rng = np.random.default_rng(0)
idx = rng.choice(len(gs), (4000, len(gs)))

print(f"\n기준 = {BASE[0]}  (RMSE {rmse(Y, base):.1f})")
print(f"{'상대 모델':24s}{'RMSE':>8s}{'ΔRMSE':>8s}{'95% CI':>20s}{'승리 fold':>11s}")
print("-" * 71)
for nm, cls, prm in OTHERS:
    p = cv(cls, prm)
    om = np.array([((Y[g == gi] - p[g == gi]) ** 2).mean() for gi in gs])
    drm = np.sqrt(om[idx].mean(1)) - np.sqrt(bm[idx].mean(1))
    lo, hi = np.percentile(drm, [2.5, 97.5])
    sig = "유의" if (lo > 0 or hi < 0) else "-"
    print(f"{nm:24s}{rmse(Y, p):8.1f}{np.sqrt(om.mean()) - np.sqrt(bm.mean()):+8.1f}"
          f"   [{lo:+6.1f},{hi:+6.1f}] {sig:4s}{(om < bm).sum():5d}/{len(gs)}")
print("-" * 71)
print("ΔRMSE < 0 이면 상대 모델이 더 정확. CI 가 0 을 포함하면 차이는 통계적으로 미검출.")
