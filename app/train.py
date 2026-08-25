"""Train_2.csv 로 Quad Ridge 모델을 학습해 app/model.json 저장.

    python app/train.py
    python app/train.py --csv data/Train_2.csv --out app/model.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrr_model import (QuadRidgeMRR, WIWNU_BANDS, cross_validate,  # noqa: E402
                       design_groups, load_csv, wiwnu)

ap = argparse.ArgumentParser(description="Quad Ridge MRR 모델 학습")
ap.add_argument("--csv", default="data/Train_2.csv")
ap.add_argument("--out", default="app/model.json")
ap.add_argument("--label", default=None, help="앱 화면에 표시할 데이터셋 이름")
ap.add_argument("--alpha", type=float, default=None,
                help="지정하지 않으면 교차검증으로 자동 선택")
a = ap.parse_args()

radius, mrr, pressure, wafer_id = load_csv(a.csv)
g = design_groups(pressure)
print(f"학습 데이터 : {a.csv}")
print(f"              웨이퍼 {len(wafer_id)}장 x 반경 {len(radius)}점, "
      f"고유 압력조건 {len(np.unique(g))}개")
print(f"              압력 범위  Z1 {pressure[:,0].min()}~{pressure[:,0].max()}  "
      f"Z2 {pressure[:,1].min()}~{pressure[:,1].max()}  "
      f"Z3 {pressure[:,2].min()}~{pressure[:,2].max()} psi")

# ---- alpha 선택 : Leave-One-Design-Out 교차검증 --------------------------
GRID = [0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
print("\nalpha 선택 (Leave-One-Design-Out 교차검증)")
print(f"  {'alpha':>7s}{'RMSE':>9s}{'중심|r|<=50':>12s}{'에지|r|>=70':>12s}")
best = None
for al in GRID:
    pred = cross_validate(pressure, mrr, radius, al)
    rmse = float(np.sqrt(((mrr - pred) ** 2).mean()))
    c = np.abs(radius) <= 50
    e = np.abs(radius) >= 70
    mark = ""
    if best is None or rmse < best[0]:
        best, mark = (rmse, al, pred), "  <-"
    print(f"  {al:7g}{rmse:9.1f}"
          f"{float(np.sqrt(((mrr[:,c]-pred[:,c])**2).mean())):12.1f}"
          f"{float(np.sqrt(((mrr[:,e]-pred[:,e])**2).mean())):12.1f}{mark}")

cv_rmse, alpha, cv_pred = best
if a.alpha is not None:
    alpha = a.alpha
    cv_pred = cross_validate(pressure, mrr, radius, alpha)
    cv_rmse = float(np.sqrt(((mrr - cv_pred) ** 2).mean()))
    print(f"\n  (--alpha {alpha} 지정됨)")

mape = float(np.mean(np.abs(cv_pred - mrr) / mrr) * 100)
print(f"\n선택된 alpha = {alpha}")
print(f"교차검증 성능 : RMSE {cv_rmse:.1f} A/min,  MAPE {mape:.2f}%")
c70 = np.abs(radius) <= 70
print(f"                edge exclusion 5mm 기준 RMSE "
      f"{float(np.sqrt(((mrr[:,c70]-cv_pred[:,c70])**2).mean())):.1f} A/min")

# ---- 반복실험 노이즈 하한 (참고) ----------------------------------------
pe = [((mrr[g == gi] - mrr[g == gi].mean(0)) ** 2).sum(0) / ((g == gi).sum() - 1)
      for gi in np.unique(g) if (g == gi).sum() > 1]
if pe:
    print(f"                (같은 조건 반복 웨이퍼로 잰 측정 재현성 한계 "
          f"{float(np.sqrt(np.mean(pe))):.1f} A/min)")

# ---- WIWNU 예측 정확도 ---------------------------------------------------
print("\nWIWNU 예측 정확도 (교차검증, 본 적 없는 압력 조건 기준)")
print(f"  {'구간':>6s}{'실측 평균':>11s}{'예측 평균':>11s}{'절대오차 평균':>14s}")
for name in WIWNU_BANDS:
    tru = np.array([wiwnu(mrr[i], radius)[name]["wiwnu_pct"] for i in range(len(mrr))])
    prd = np.array([wiwnu(cv_pred[i], radius)[name]["wiwnu_pct"] for i in range(len(mrr))])
    print(f"  {name:>6s}{tru.mean():10.2f}%{prd.mean():10.2f}%"
          f"{np.abs(prd - tru).mean():13.2f}%p")

# ---- 최종 학습 & 저장 ----------------------------------------------------
model = QuadRidgeMRR(alpha).fit(pressure, mrr, radius)
model.sigma_ = (mrr - cv_pred).std(0)          # 반경별 예측 불확실도
model.cv_ = {"rmse": cv_rmse, "mape": mape, "alpha": alpha,
             "n_wafer": int(len(wafer_id)), "source": str(a.csv),
             "label": a.label or Path(a.csv).stem}
model.save(a.out)
print(f"\n저장 완료 : {a.out}")
print(f"실행      : python app/predict.py 2.0 2.2 2.0 --model {a.out}")
