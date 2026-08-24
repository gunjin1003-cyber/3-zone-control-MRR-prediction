"""배포용 예측기: 모델 + 링압력 보간 + 반경별 불확실도 + 외삽 경고."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import models as M
from .cv import nested_cv, rmse
from .data import CMPData


@dataclass
class MRRPredictor:
    model_cls: type = M.SymmetryRidge
    params: dict = field(default_factory=dict)

    # 학습 후 채워짐
    model_: object = None
    radius_: np.ndarray = None
    sigma_: np.ndarray = None       # 반경별 CV 잔차 표준편차
    ring_coef_: np.ndarray = None   # ring ~ zones 보간식
    ring_rmse_: float = 0.0
    train_P_: np.ndarray = None
    cv_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fit(self, data: CMPData, cv_groups: np.ndarray | None = None):
        P, Y, r = data.pressure, data.mrr, data.radius
        self.radius_ = r
        self.train_P_ = P.copy()

        # 링 압력 보간식 (사용자가 링 압력을 모를 때 사용)
        X = np.c_[np.ones(len(P)), P[:, :3]]
        self.ring_coef_, *_ = np.linalg.lstsq(X, P[:, 3], rcond=None)
        self.ring_rmse_ = float(np.sqrt(np.mean((X @ self.ring_coef_ - P[:, 3]) ** 2)))

        # 교차검증으로 반경별 불확실도 추정
        g = data.design_groups() if cv_groups is None else cv_groups
        pred, _ = nested_cv(self.model_cls, {k: [v] for k, v in self.params.items()},
                            P, Y, r, g, inner_groups=g)
        self.sigma_ = (Y - pred).std(0)
        self.cv_ = dict(rmse=rmse(Y, pred),
                        mape=float(np.mean(np.abs(pred - Y) / Y) * 100))

        # 전체 데이터로 최종 적합
        m = self.model_cls(**self.params)
        try:
            m.fit(P, Y, radius=r)
        except TypeError:
            m.fit(P, Y)
        self.model_ = m
        return self

    # ------------------------------------------------------------------
    def _prep(self, z1, z2, z3, ring=None):
        z = np.atleast_2d(np.column_stack(np.broadcast_arrays(z1, z2, z3)))
        if ring is None:
            rg = np.c_[np.ones(len(z)), z] @ self.ring_coef_
        else:
            rg = np.broadcast_to(np.asarray(ring, float), (len(z),))
        return np.c_[z, rg]

    def predict(self, z1, z2, z3, ring=None) -> np.ndarray:
        """(n,R) 예측 MRR 프로파일."""
        return self.model_.predict(self._prep(z1, z2, z3, ring))

    def predict_interval(self, z1, z2, z3, ring=None, z=1.96):
        mu = self.predict(z1, z2, z3, ring)
        return mu, mu - z * self.sigma_, mu + z * self.sigma_

    # ------------------------------------------------------------------
    def extrapolation_score(self, z1, z2, z3) -> np.ndarray:
        """학습 설계점까지의 최소 거리(psi). 0.3 이상이면 외삽 주의."""
        q = np.atleast_2d(np.column_stack(np.broadcast_arrays(z1, z2, z3)))
        return np.min(np.linalg.norm(q[:, None, :] - self.train_P_[None, :, :3],
                                     axis=2), axis=1)

    # ------------------------------------------------------------------
    @staticmethod
    def uniformity(profile: np.ndarray) -> dict:
        p = np.atleast_2d(profile)
        mean = p.mean(1)
        return dict(mean=mean,
                    std=p.std(1, ddof=1),
                    nu_pct=p.std(1, ddof=1) / mean * 100,
                    range_pct=(p.max(1) - p.min(1)) / mean * 100,
                    min=p.min(1), max=p.max(1))

    # ------------------------------------------------------------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "MRRPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
