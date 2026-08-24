"""하이퍼파라미터를 고정한 공정 비교 + fold 간 표준오차(차이의 유의성 판단용)."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import folds, rmse, region_report

d = load("data/Train_1.csv")
P, Y, r, g = d.pressure, d.mrr, d.radius, d.design_groups()

CONFIGS = [
    ("평균 프로파일(기준선)",       M.MeanProfile,      {}),
    ("Preston 스케일(기준선)",      M.PrestonScale,     {}),
    ("반경별 능형 [zone]",          M.RadialRidge,      dict(alpha=1.0, feat="zone")),
    ("반경별 능형 [zone+ring]",     M.RadialRidge,      dict(alpha=1.0, feat="zone+ring")),
    ("반경별 능형 [quad]",          M.RadialRidge,      dict(alpha=3.0, feat="quad")),
    ("반경별 능형 [phys]",          M.RadialRidge,      dict(alpha=3.0, feat="phys")),
    ("저계수 능형 rank3",           M.ReducedRankRidge, dict(alpha=1.0, feat="zone+ring", rank=3)),
    ("PLS 2성분 [zone+ring]",       M.SklearnMulti,     dict(kind="pls", feat="zone+ring", n_comp=2)),
    ("PLS 3성분 [quad]",            M.SklearnMulti,     dict(kind="pls", feat="quad", n_comp=3)),
    ("대칭성 능형 [zone+ring]",     M.SymmetryRidge,    dict(alpha=1.0, feat="zone+ring", rank=1, alpha_a=1.0)),
    ("대칭성 능형 [quad]",          M.SymmetryRidge,    dict(alpha=1.0, feat="quad", rank=1, alpha_a=1.0)),
    ("대칭성 PLS [zone+ring]",      M.SymmetryPLS,      dict(n_comp=2, feat="zone+ring", rank=1)),
    ("GPR 다출력",                  M.SklearnMulti,     dict(kind="gpr", feat="zone+ring")),
    ("랜덤포레스트",                M.SklearnMulti,     dict(kind="rf", feat="zone+ring")),
    ("신경망 MLP",                  M.SklearnMulti,     dict(kind="mlp", feat="zone+ring", alpha=1.0)),
]

rows = []
for label, cls, prm in CONFIGS:
    pred = np.zeros_like(Y)
    for tr, te in folds(g):
        m = cls(**prm)
        try:
            m.fit(P[tr], Y[tr], radius=r)
        except TypeError:
            m.fit(P[tr], Y[tr])
        pred[te] = m.predict(P[te])
    # fold(설계조건) 단위 MSE -> 부트스트랩 표준오차
    fold_mse = np.array([((Y[g == gi] - pred[g == gi]) ** 2).mean() for gi in np.unique(g)])
    rng = np.random.default_rng(0)
    bs = np.sqrt(rng.choice(fold_mse, (2000, len(fold_mse)), replace=True).mean(1))
    rows.append(dict(label=label, rmse=rmse(Y, pred), se=float(bs.std()),
                     mape=float(np.mean(np.abs(pred - Y) / Y) * 100),
                     region=region_report(r, Y, pred)))
    np.save(f"reports/fixedpred_{len(rows)}.npy", pred)

rows.sort(key=lambda x: x["rmse"])
print(f"{'모델(하이퍼파라미터 고정)':32s}{'RMSE':>8s}{'±SE':>7s}{'MAPE%':>8s}{'중심':>8s}{'중간':>7s}{'에지':>8s}")
print("-" * 79)
for x in rows:
    q = x["region"]
    print(f"{x['label']:32s}{x['rmse']:8.1f}{x['se']:7.1f}{x['mape']:8.2f}"
          f"{q['center |r|<=50']:8.1f}{q['mid 55-68']:7.1f}{q['edge |r|>=70']:8.1f}")
print("-" * 79)
print(f"{'반복실험 순수오차 (하한)':32s}{118.4:8.1f}{'':7s}{'':8s}{40.0:8.1f}{46.0:7.1f}{162.0:8.1f}")

best = rows[0]["rmse"]
print(f"\n최고 모델 대비 1 SE(={rows[0]['se']:.0f}) 이내 = 통계적으로 구분 불가:")
for x in rows:
    if x["rmse"] - best <= rows[0]["se"]:
        print(f"   {x['label']}  ({x['rmse']:.1f})")

json.dump(rows, open("reports/benchmark_fixed.json", "w"), indent=2, ensure_ascii=False)
