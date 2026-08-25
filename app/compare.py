"""두 학습 데이터셋(Train_1 21장 / Train_2 18장) 모델 비교.

    python app/compare.py                       # 기본 조건표
    python app/compare.py 2.0 2.2 2.0           # 특정 압력 한 건 상세 비교
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrr_model import QuadRidgeMRR, wiwnu  # noqa: E402

HERE = Path(__file__).resolve().parent
A = QuadRidgeMRR.load(HERE / "model_train1.json")     # Train_1, 21장
B = QuadRidgeMRR.load(HERE / "model.json")            # Train_2, 18장
r = A.radius_
assert np.allclose(r, B.radius_), "두 모델의 반경 격자가 다릅니다."

print("=" * 78)
print(f"{'':22s}{'Train_1 (21장)':>18s}{'Train_2 (18장)':>18s}{'차이':>12s}")
print("-" * 78)
for key, lab, fmt in [("n_wafer", "학습 웨이퍼", "{:.0f}"),
                      ("alpha", "선택된 alpha", "{:.2f}"),
                      ("rmse", "교차검증 RMSE", "{:.1f}"),
                      ("mape", "교차검증 MAPE %", "{:.2f}")]:
    a, b = A.cv_.get(key, 0), B.cv_.get(key, 0)
    print(f"{lab:22s}{fmt.format(a):>18s}{fmt.format(b):>18s}{b - a:+12.2f}")
print("-" * 78)
print("Train_1 은 Train_2 에 웨이퍼 19,20,21 (2.7, 3.2, 3.0 psi 반복 3장) 이 추가된 것")

if len(sys.argv) >= 4:
    z = [float(v) for v in sys.argv[1:4]]
    ya, yb = A.predict_one(*z), B.predict_one(*z)
    wa, wb = wiwnu(ya, r), wiwnu(yb, r)
    print("\n" + "=" * 78)
    print(f"압력  Zone1={z[0]:g}  Zone2={z[1]:g}  Zone3={z[2]:g} psi")
    print("-" * 78)
    print(f"{'':16s}{'Train_1':>14s}{'Train_2':>14s}{'차이':>12s}")
    print(f"{'평균 MRR':16s}{wa['all']['mean']:14.1f}{wb['all']['mean']:14.1f}"
          f"{wb['all']['mean'] - wa['all']['mean']:+12.1f}")
    for nm in ("all", "5mm", "7mm"):
        print(f"{'WIWNU ' + nm:16s}{wa[nm]['wiwnu_pct']:13.2f}%{wb[nm]['wiwnu_pct']:13.2f}%"
              f"{wb[nm]['wiwnu_pct'] - wa[nm]['wiwnu_pct']:+11.2f}%p")
    print(f"\n프로파일 차이  RMS {float(np.sqrt(((ya - yb) ** 2).mean())):.1f} A/min, "
          f"최대 {float(np.abs(ya - yb).max()):.1f} A/min")
    print(f"{'r (mm)':>8s}{'Train_1':>12s}{'Train_2':>12s}{'차이':>10s}")
    for i in np.argsort(-r):
        print(f"{r[i]:8.0f}{ya[i]:12.1f}{yb[i]:12.1f}{yb[i] - ya[i]:+10.1f}")
    sys.exit(0)

GRID = [(2.0, 2.0, 2.0), (2.0, 2.2, 2.0), (1.5, 2.0, 2.0), (2.5, 2.5, 2.5),
        (2.0, 2.0, 2.5), (2.0, 1.5, 2.0), (2.7, 3.2, 3.0), (3.0, 3.0, 3.0)]
print(f"\n{'Z1':>5s}{'Z2':>5s}{'Z3':>5s} |{'평균MRR T1':>11s}{'T2':>9s}"
      f" |{'WIWNU5 T1':>10s}{'T2':>8s} |{'WIWNU7 T1':>10s}{'T2':>8s} |{'프로파일차 RMS':>14s}")
print("-" * 92)
for z in GRID:
    ya, yb = A.predict_one(*z), B.predict_one(*z)
    wa, wb = wiwnu(ya, r), wiwnu(yb, r)
    print(f"{z[0]:5g}{z[1]:5g}{z[2]:5g} |{wa['all']['mean']:11.1f}{wb['all']['mean']:9.1f}"
          f" |{wa['5mm']['wiwnu_pct']:9.2f}%{wb['5mm']['wiwnu_pct']:7.2f}%"
          f" |{wa['7mm']['wiwnu_pct']:9.2f}%{wb['7mm']['wiwnu_pct']:7.2f}%"
          f" |{float(np.sqrt(((ya - yb) ** 2).mean())):14.1f}")
print("-" * 92)
print("마지막 두 조건(2.7/3.2/3.0, 3.0/3.0/3.0)은 Train_2 에는 반복 데이터가 없어 차이가 커진다.")
