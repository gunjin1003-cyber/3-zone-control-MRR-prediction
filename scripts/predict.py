"""압력 -> MRR 프로파일 예측 CLI.

예)  python scripts/predict.py 2.0 2.2 2.0
     python scripts/predict.py 2.7 3.2 3.0 --ring 3.01 --plot out.png --csv out.csv
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.pipeline import MRRPredictor

ap = argparse.ArgumentParser(description="3-zone 압력으로 SiO2 CMP MRR 프로파일 예측")
ap.add_argument("z1", type=float, help="Zone 1 압력 [psi] (최외곽 에지)")
ap.add_argument("z2", type=float, help="Zone 2 압력 [psi] (중간 링)")
ap.add_argument("z3", type=float, help="Zone 3 압력 [psi] (중심)")
ap.add_argument("--ring", type=float, default=None, help="리테이너 링 압력 [psi] (미지정 시 자동 보간)")
ap.add_argument("--model", default="models/mrr_model.pkl")
ap.add_argument("--csv", default=None, help="예측 결과 CSV 저장 경로")
ap.add_argument("--plot", default=None, help="프로파일 그림 저장 경로")
a = ap.parse_args()

p = MRRPredictor.load(a.model)
mu, lo, hi = p.predict_interval(a.z1, a.z2, a.z3, a.ring)
mu, lo, hi = mu[0], lo[0], hi[0]
r = p.radius_
u = {k: float(np.atleast_1d(v)[0]) for k, v in p.uniformity(mu).items()}
ring = float(p._prep(a.z1, a.z2, a.z3, a.ring)[0, 3])
ext = float(p.extrapolation_score(a.z1, a.z2, a.z3)[0])

print(f"\n입력 압력 : Zone1={a.z1}  Zone2={a.z2}  Zone3={a.z3}  "
      f"Ring={ring:.2f}{'' if a.ring is not None else ' (자동 보간)'} psi")
print(f"모델 CV   : RMSE {p.cv_['rmse']:.1f} A/min ({p.cv_['mape']:.1f}%)")
if ext > 0.3:
    print(f"[경고] 학습 설계점에서 {ext:.2f} psi 떨어진 외삽 영역입니다. 예측 신뢰도가 낮습니다.")

print("\n  r(mm)      MRR    95% 구간")
print("  " + "-" * 38)
for i in range(len(r)):
    print(f"  {r[i]:6.0f}  {mu[i]:8.1f}   [{lo[i]:7.1f}, {hi[i]:7.1f}]")
print("  " + "-" * 38)
print(f"\n  평균 MRR      : {u['mean']:8.1f} A/min")
print(f"  표준편차      : {u['std']:8.1f} A/min")
print(f"  불균일도(1s)  : {u['nu_pct']:8.2f} %")
print(f"  Range/mean    : {u['range_pct']:8.2f} %")
print(f"  최소 / 최대   : {u['min']:8.1f} / {u['max']:.1f}")

if a.csv:
    import csv
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["radius_mm", "MRR_pred", "lo95", "hi95"])
        w.writerows(zip(r, mu.round(2), lo.round(2), hi.round(2)))
    print(f"\nCSV 저장: {a.csv}")

if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = np.argsort(r)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(r[o], lo[o], hi[o], alpha=.25, color="C0", label="95% 예측구간")
    ax.plot(r[o], mu[o], "-o", ms=3, color="C0", label="예측 MRR")
    ax.axhline(u["mean"], ls="--", c="gray", lw=.8)
    ax.set_xlabel("Radius [mm]"); ax.set_ylabel("MRR [A/min]")
    ax.set_title(f"Z1={a.z1}, Z2={a.z2}, Z3={a.z3}, Ring={ring:.2f} psi"
                 f"   |  mean={u['mean']:.0f}, NU={u['nu_pct']:.2f}%")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(a.plot, dpi=130)
    print(f"그림 저장: {a.plot}")
