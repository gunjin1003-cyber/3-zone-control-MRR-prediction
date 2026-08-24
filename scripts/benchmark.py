"""모델 후보 벤치마크: 중첩 LOGO-CV(정직) + LOWO-CV(참고)."""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import nested_cv, rmse, region_report

d = load("data/Train_1.csv")
P, Y, r = d.pressure, d.mrr, d.radius
dg = d.design_groups()                 # 동일 압력 조건 = 같은 그룹
wg = np.arange(d.n_wafer)              # 웨이퍼 단위

A = [0.01, 0.1, 1, 3, 10, 30, 100]
FEATS = ["zone", "zone+ring", "quad", "quad+ring", "phys"]

CANDIDATES = [
    ("0. 평균 프로파일(기준선)",      M.MeanProfile,   {}),
    ("0. Preston 스케일(기준선)",     M.PrestonScale,  {}),
    ("A. 반경별 능형회귀(선형)",       M.RadialRidge,   dict(alpha=A, feat=["zone", "zone+ring"], smooth=[0.0], log=[False])),
    ("B. 반경별+2차/교호",            M.RadialRidge,   dict(alpha=A, feat=["quad", "quad+ring", "phys"], smooth=[0.0], log=[False])),
    ("C. 반경별+계수평활",            M.RadialRidge,   dict(alpha=A, feat=FEATS, smooth=[0.0, 2.0, 4.0, 8.0], log=[False])),
    ("D. 반경별 로그(곱셈형)",         M.RadialRidge,   dict(alpha=A, feat=FEATS, smooth=[0.0, 2.0, 4.0], log=[True])),
    ("E. PCA + 능형회귀",             M.PCARegressor,  dict(k=[2, 3, 4, 5], alpha=A, feat=["zone", "zone+ring", "quad"], head=["ridge"])),
    ("F. PCA + 가우시안과정(GPR)",     M.PCARegressor,  dict(k=[2, 3, 4], feat=["zone", "zone+ring"], head=["gpr"])),
    ("G. GPR 다출력",                 M.SklearnMulti,  dict(kind=["gpr"], feat=["zone", "zone+ring"])),
    ("H. 커널능형회귀(RBF)",          M.SklearnMulti,  dict(kind=["krr"], feat=["zone", "zone+ring"], alpha=[0.01, 0.1, 1], gamma=[0.05, 0.2, 1.0])),
    ("I. PLS",                        M.SklearnMulti,  dict(kind=["pls"], feat=["zone", "zone+ring", "quad"], n_comp=[1, 2, 3])),
    ("J. 랜덤포레스트",               M.SklearnMulti,  dict(kind=["rf"], feat=["zone", "zone+ring"], leaf=[1, 2])),
    ("K. Extra Trees",                M.SklearnMulti,  dict(kind=["et"], feat=["zone", "zone+ring"], leaf=[1, 2])),
    ("L. 그래디언트 부스팅",           M.SklearnMulti,  dict(kind=["gbr"], feat=["zone", "zone+ring"])),
    ("M. 신경망(MLP)",                M.SklearnMulti,  dict(kind=["mlp"], feat=["zone", "zone+ring"], alpha=[0.1, 1, 10], hidden=[(32, 32), (64, 64)])),
    ("N. 대칭성인지 능형회귀",         M.SymmetryRidge, dict(alpha=A, feat=FEATS, rank=[0, 1, 2, 3], alpha_a=[0.1, 1, 10], smooth=[0.0])),
    ("O. 대칭성인지+계수평활",         M.SymmetryRidge, dict(alpha=A, feat=["zone", "zone+ring", "quad"], rank=[1, 2], alpha_a=[1.0], smooth=[0.0, 2.0, 4.0])),
    ("P. 대칭성인지 + 잔차GP",         M.RidgePlusGP,   dict(alpha=[0.1, 1, 10], feat=["zone", "zone+ring"], rank=[1], k=[1, 2], smooth=[0.0])),
]

rows = []
for label, cls, space in CANDIDATES:
    t0 = time.time()
    pd_, chosen = nested_cv(cls, space, P, Y, r, dg, inner_groups=dg)
    pw_, _ = nested_cv(cls, space, P, Y, r, wg, inner_groups=dg)
    rows.append(dict(label=label, design=rmse(Y, pd_), wafer=rmse(Y, pw_),
                     mape=float(np.mean(np.abs(pd_ - Y) / Y) * 100),
                     region=region_report(r, Y, pd_),
                     chosen=chosen, sec=time.time() - t0))
    np.save(f"reports/pred_{label.split('.')[0]}.npy", pd_)
    print(f"  {label:28s} design {rows[-1]['design']:7.1f} | wafer {rows[-1]['wafer']:7.1f}"
          f" | {rows[-1]['sec']:5.1f}s", flush=True)

rows.sort(key=lambda x: x["design"])
print("\n" + "=" * 92)
print(f"{'모델':32s}{'LOGO-CV':>9s}{'LOWO-CV':>9s}{'MAPE%':>7s}   {'중심':>7s}{'중간':>7s}{'에지':>8s}")
print("-" * 92)
for x in rows:
    g = x["region"]
    print(f"{x['label']:32s}{x['design']:9.1f}{x['wafer']:9.1f}{x['mape']:7.2f}   "
          f"{g['center |r|<=50']:7.1f}{g['mid 55-68']:7.1f}{g['edge |r|>=70']:8.1f}")
print("-" * 92)
print(f"{'*** 반복실험 순수오차(하한)':32s}{118.4:9.1f}{'':9s}{'':7s}   {40:7.1f}{'':7s}{162:8.1f}")
print("=" * 92)

with open("reports/benchmark.json", "w") as f:
    json.dump([{k: v for k, v in x.items() if k != "chosen"} for x in rows], f,
              indent=2, ensure_ascii=False)

best = rows[0]
from collections import Counter
print("\n최고 모델:", best["label"])
print("fold 별 선택된 하이퍼파라미터 빈도:")
for p, c in Counter(json.dumps(x, sort_keys=True) for x in best["chosen"]).most_common():
    print(f"   {c:2d}x  {p}")
