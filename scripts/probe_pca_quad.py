"""PCA + 2차(Quad) Ridge 조합 검증.

질문: 프로파일을 PCA 로 압축한 뒤 2차 압력항으로 능형회귀하면 더 좋아지는가?
답  : 정확도는 상위권과 동률이지만, PCA 절단은 k>=4 에서 아무 일도 하지 않는다.
      규제는 전적으로 ridge 의 alpha 가 담당한다.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import folds, nested_cv, rmse, region_report

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


print("=" * 62)
print("1) k x alpha 격자")
print(f"{'k':>3s}{'alpha':>8s}{'RMSE':>9s}{'중심':>8s}{'중간':>8s}{'에지':>9s}")
for k in (2, 3, 4, 5, 6):
    for al in (0.1, 1, 3, 10):
        p = cv(M.PCARegressor, dict(k=k, head="ridge", alpha=al, feat="quad"))
        q = region_report(r, Y, p)
        print(f"{k:3d}{al:8g}{rmse(Y, p):9.1f}{q['center |r|<=50']:8.1f}"
              f"{q['mid 55-68']:8.1f}{q['edge |r|>=70']:9.1f}")

print("=" * 62)
print("2) PCA 절단이 실제로 일을 하는가 (alpha=1 고정)")
for k in (2, 3, 4, 5, 6, 10, 20):
    print(f"   k={k:2d} : {rmse(Y, cv(M.PCARegressor, dict(k=k, head='ridge', alpha=1.0, feat='quad'))):6.1f}")
print(f"   PCA 없음(= 순수 quad ridge) : "
      f"{rmse(Y, cv(M.RadialRidge, dict(alpha=1.0, feat='quad'))):6.1f}")
print("   -> k>=5 에서 PCA 없는 경우와 완전히 일치. 절단이 무효.")

print("=" * 62)
print("3) 중첩 CV — fold 안에서 k 를 고르게 하면 무엇을 고르는가")
p, ch = nested_cv(M.PCARegressor, dict(k=[2, 3, 4, 5, 6], alpha=[0.1, 0.3, 1, 3, 10],
                                       feat=["quad"], head=["ridge"]),
                  P, Y, r, g, inner_groups=g)
from collections import Counter
import json
print(f"   중첩 CV RMSE = {rmse(Y, p):.1f}")
for q, c in Counter(json.dumps(x, sort_keys=True) for x in ch).most_common():
    print(f"   {c:2d}x  {q}")
print("   -> 내부 CV 가 스스로 k=5~6 (= 절단 해제) 을 고른다.")
