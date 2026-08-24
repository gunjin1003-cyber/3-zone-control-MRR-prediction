"""교차검증 하네스 (하이퍼파라미터 내부 선택을 포함한 중첩 CV)."""
from __future__ import annotations

import itertools
import numpy as np


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def folds(groups: np.ndarray):
    """leave-one-group-out 인덱스 생성."""
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        yield tr, te


def grid(space: dict):
    if not space:
        yield {}
        return
    keys = list(space)
    for vals in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, vals))


def _fit(model_cls, params, P, Y, radius):
    m = model_cls(**params)
    try:
        m.fit(P, Y, radius=radius)
    except TypeError:
        m.fit(P, Y)
    return m


def nested_cv(model_cls, space, P, Y, radius, groups, inner_groups=None):
    """외부 LOGO + 내부 LOGO 로 하이퍼파라미터를 고른 정직한 CV.

    Returns
    -------
    pred : (n,R)  각 웨이퍼가 검증 fold 였을 때의 예측
    chosen : 각 fold 에서 선택된 파라미터
    """
    inner_groups = groups if inner_groups is None else inner_groups
    pred = np.zeros_like(Y)
    chosen = []
    space_list = list(grid(space))

    for tr, te in folds(groups):
        if len(space_list) == 1:
            best = space_list[0]
        else:
            gi = inner_groups[tr]
            scores = []
            for params in space_list:
                errs = []
                for itr, ite in folds(gi):
                    m = _fit(model_cls, params, P[tr][itr], Y[tr][itr], radius)
                    errs.append(((m.predict(P[tr][ite]) - Y[tr][ite]) ** 2).ravel())
                scores.append(np.mean(np.concatenate(errs)))
            best = space_list[int(np.argmin(scores))]
        chosen.append(best)
        m = _fit(model_cls, best, P[tr], Y[tr], radius)
        pred[te] = m.predict(P[te])
    return pred, chosen


def region_report(radius, Y, pred):
    """중심/중간/에지 구간별 RMSE."""
    a = np.abs(radius)
    reg = {"center |r|<=50": a <= 50,
           "mid 55-68": (a >= 55) & (a <= 68),
           "edge |r|>=70": a >= 70}
    return {k: rmse(Y[:, m], pred[:, m]) for k, m in reg.items()}
