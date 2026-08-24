"""라운드 2: 선두 모델군(저차원 잠재구조 + 대칭성)을 정밀 비교."""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import nested_cv, rmse, region_report

d = load("data/Train_1.csv")
P, Y, r = d.pressure, d.mrr, d.radius
dg, wg = d.design_groups(), np.arange(d.n_wafer)

A = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]
FEATS = ["zone", "zone+ring", "quad", "quad+ring", "phys"]

CAND = [
    ("I+ PLS (정밀)",             M.SklearnMulti,     dict(kind=["pls"], feat=FEATS, n_comp=[1, 2, 3, 4])),
    ("Q. 저계수 능형회귀",         M.ReducedRankRidge, dict(alpha=A, feat=FEATS, rank=[1, 2, 3, 4])),
    ("N+ 대칭성 능형(정밀)",       M.SymmetryRidge,    dict(alpha=A, feat=FEATS, rank=[0, 1, 2], alpha_a=[0.1, 1, 10], smooth=[0.0, 2.0])),
    ("R. 대칭성 + PLS",           M.SymmetryPLS,      dict(n_comp=[1, 2, 3, 4], feat=FEATS, rank=[0, 1, 2], alpha_a=[1.0])),
]

rows = []
for label, cls, space in CAND:
    t0 = time.time()
    pd_, chosen = nested_cv(cls, space, P, Y, r, dg, inner_groups=dg)
    pw_, _ = nested_cv(cls, space, P, Y, r, wg, inner_groups=dg)
    rows.append(dict(label=label, design=rmse(Y, pd_), wafer=rmse(Y, pw_),
                     mape=float(np.mean(np.abs(pd_ - Y) / Y) * 100),
                     region=region_report(r, Y, pd_), chosen=chosen))
    g = rows[-1]["region"]
    print(f"{label:24s} design {rows[-1]['design']:7.1f} | wafer {rows[-1]['wafer']:7.1f} "
          f"| center {g['center |r|<=50']:6.1f} mid {g['mid 55-68']:6.1f} edge {g['edge |r|>=70']:7.1f}"
          f" | {time.time()-t0:5.1f}s", flush=True)

from collections import Counter
for x in sorted(rows, key=lambda z: z["design"]):
    print(f"\n{x['label']}  (LOGO {x['design']:.1f})  선택 빈도:")
    for p, c in Counter(json.dumps(y, sort_keys=True) for y in x["chosen"]).most_common(3):
        print(f"   {c:2d}x  {p}")

json.dump([{k: v for k, v in x.items() if k != "chosen"} for x in rows],
          open("reports/benchmark2.json", "w"), indent=2, ensure_ascii=False)
