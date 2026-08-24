"""전체 데이터에 대해 LOGO-CV 로 최종 하이퍼파라미터 확정 -> reports/best_config.json"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import folds, grid, rmse

d = load("data/Train_1.csv")
P, Y, r, g = d.pressure, d.mrr, d.radius, d.design_groups()

MODEL = M.SymmetryRidge
SPACE = dict(alpha=[0.01, 0.1, 0.3, 1, 3, 10, 30, 100],
             feat=["zone", "zone+ring", "quad", "quad+ring", "phys"],
             rank=[0, 1, 2, 3],
             alpha_a=[0.1, 1, 10],
             smooth=[0.0])

best, best_s = None, np.inf
for params in grid(SPACE):
    errs = []
    for tr, te in folds(g):
        m = MODEL(**params).fit(P[tr], Y[tr], radius=r)
        errs.append(((m.predict(P[te]) - Y[te]) ** 2).ravel())
    s = np.sqrt(np.mean(np.concatenate(errs)))
    if s < best_s:
        best, best_s = params, s

print("최종 선택:", best, f"  LOGO-CV RMSE = {best_s:.1f} A/min")
pathlib.Path("reports").mkdir(exist_ok=True)
json.dump({"model": MODEL.__name__, "params": best}, open("reports/best_config.json", "w"),
          indent=2, ensure_ascii=False)
print("reports/best_config.json 저장")
