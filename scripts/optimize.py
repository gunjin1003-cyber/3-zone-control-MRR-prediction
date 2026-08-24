"""역방향: 목표(균일도 최대 / 목표 프로파일 추종)를 만족하는 3-zone 압력 탐색.

예)  python scripts/optimize.py --mode uniform --target-mean 1200
     python scripts/optimize.py --mode profile --target-csv my_target.csv
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import differential_evolution
from mrrpred.pipeline import MRRPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["uniform", "profile"], default="uniform")
ap.add_argument("--target-mean", type=float, default=None, help="목표 평균 MRR [A/min]")
ap.add_argument("--target-csv", default=None, help="radius,MRR 2열 CSV (mode=profile)")
ap.add_argument("--bounds", default="1.3,3.2", help="압력 탐색 범위 lo,hi [psi]")
ap.add_argument("--edge-exclude", type=float, default=0.0,
                help="이 반경(mm) 초과 지점은 목적함수에서 제외 (edge exclusion)")
ap.add_argument("--model", default="models/mrr_model.pkl")
a = ap.parse_args()

p = MRRPredictor.load(a.model)
r = p.radius_
use = np.abs(r) <= (a.edge_exclude if a.edge_exclude > 0 else np.inf)

target = None
if a.mode == "profile":
    if not a.target_csv:
        ap.error("--mode profile 은 --target-csv 필요")
    d = np.genfromtxt(a.target_csv, delimiter=",", skip_header=1)
    target = np.interp(r, d[:, 0][np.argsort(d[:, 0])], d[:, 1][np.argsort(d[:, 0])])

lo, hi = (float(x) for x in a.bounds.split(","))


def cost(z):
    y = p.predict(z[0], z[1], z[2])[0][use]
    if a.mode == "profile":
        c = np.sqrt(np.mean((y - target[use]) ** 2))
    else:
        c = y.std(ddof=1) / y.mean() * 100                     # 불균일도 %
        if a.target_mean:
            c += 0.05 * abs(y.mean() - a.target_mean) / a.target_mean * 100
    # 학습 영역 밖으로 크게 벗어나면 벌점
    return c + 30.0 * max(0.0, p.extrapolation_score(*z)[0] - 0.4)


res = differential_evolution(cost, [(lo, hi)] * 3, seed=0, tol=1e-8,
                             maxiter=400, polish=True)
z = res.x
y = p.predict(*z)[0]
u = {k: float(np.atleast_1d(v)[0]) for k, v in p.uniformity(y[use]).items()}
ext = p.extrapolation_score(*z)[0]

print(f"\n최적 압력  : Zone1={z[0]:.2f}  Zone2={z[1]:.2f}  Zone3={z[2]:.2f} psi")
print(f"목적함수   : {res.fun:.4f}")
print(f"예상 평균 MRR : {u['mean']:.1f} A/min")
print(f"예상 불균일도 : {u['nu_pct']:.2f} % (1s){'  [|r|<=%g mm 기준]' % a.edge_exclude if a.edge_exclude else ''}")
print(f"Range/mean    : {u['range_pct']:.2f} %")
print(f"외삽 거리     : {ext:.2f} psi" + ("  <-- 학습영역 밖, 검증 실험 권장" if ext > 0.3 else ""))
print("\n  r(mm)     MRR")
for i in np.argsort(-r):
    print(f"  {r[i]:6.0f}  {y[i]:8.1f}")
