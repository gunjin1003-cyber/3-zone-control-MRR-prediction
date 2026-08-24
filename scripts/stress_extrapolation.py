"""외삽 스트레스 테스트: 저압군/고압군을 통째로 나눠 서로를 예측."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import rmse

d = load("data/Train_1.csv")
P, Y, r = d.pressure, d.mrr, d.radius
low = np.arange(15)          # 1.3 ~ 2.5 psi
high = np.arange(15, 21)     # 2.7 ~ 3.2 psi

CONFIGS = [
    ("대칭성 능형 [quad]",   M.SymmetryRidge, dict(alpha=1.0, feat="quad", rank=1, alpha_a=1.0)),
    ("반경별 능형 [zone]",   M.RadialRidge,   dict(alpha=1.0, feat="zone")),
    ("반경별 능형 [quad]",   M.RadialRidge,   dict(alpha=3.0, feat="quad")),
    ("PLS 3성분 [quad]",     M.SklearnMulti,  dict(kind="pls", feat="quad", n_comp=3)),
    ("GPR 다출력",           M.SklearnMulti,  dict(kind="gpr", feat="zone")),
    ("랜덤포레스트",          M.SklearnMulti,  dict(kind="rf", feat="zone")),
]

print("압력 공간이 1.3-2.5 / 2.7-3.2 두 덩어리로 갈라져 있어, 한쪽만 보고 다른 쪽을")
print("맞히는 것은 순수 외삽입니다. 실사용에서 학습범위를 벗어날 때의 위험도 측정.\n")
print(f"{'모델':26s}{'저압->고압':>12s}{'고압->저압':>12s}{'평균 MRR 편향(저->고)':>22s}")
print("-" * 72)
for label, cls, prm in CONFIGS:
    out = []
    bias = None
    for tr, te in [(low, high), (high, low)]:
        m = cls(**prm)
        try:
            m.fit(P[tr], Y[tr], radius=r)
        except TypeError:
            m.fit(P[tr], Y[tr])
        pr = m.predict(P[te])
        out.append(rmse(Y[te], pr))
        if bias is None:
            bias = float(pr.mean() - Y[te].mean())
    print(f"{label:26s}{out[0]:12.1f}{out[1]:12.1f}{bias:22.1f}")
print("-" * 72)
print(f"{'참고: 고압군 평균 MRR':26s}{Y[high].mean():12.1f}")
print(f"{'참고: 저압군 평균 MRR':26s}{'':12s}{Y[low].mean():12.1f}")
