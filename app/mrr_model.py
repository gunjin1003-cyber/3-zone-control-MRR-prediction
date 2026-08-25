"""SiO2 CMP 3-Zone 압력 -> MRR 프로파일 예측 (Quad Ridge).

의존성: numpy 만 필요.

모델
----
반경 r 마다 독립적으로 능형회귀(ridge)를 적합한다.

    MRR(r) = a(r) + sum_j B_j(r) * f_j(P)

    f(P) = [Z1, Z2, Z3, Z1^2, Z2^2, Z3^2, Z1*Z2, Z1*Z3, Z2*Z3]   (2차 = quad)

- 특징은 학습셋 평균/표준편차로 표준화한 뒤 사용한다.
- 절편에는 규제를 걸지 않고, 기울기 계수에만 alpha 를 건다.
- 학습 결과(계수, 표준화 상수, 반경 격자)는 JSON 한 파일로 저장한다.

WIWNU
-----
    WIWNU = (표본표준편차 / 평균) * 100        [%]

    all  : -74 ~ +74 mm  (측정 전 구간, 43점)
    5mm  : -70 ~ +70 mm  (edge exclusion 5 mm, 35점)
    7mm  : -68 ~ +68 mm  (edge exclusion 7 mm, 33점)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_NAMES = ["Z1", "Z2", "Z3", "Z1^2", "Z2^2", "Z3^2", "Z1*Z2", "Z1*Z3", "Z2*Z3"]

# WIWNU 구간 정의: 이름 -> 포함할 최대 |r| [mm]
WIWNU_BANDS = {"all": 74.0, "5mm": 70.0, "7mm": 68.0}


# ---------------------------------------------------------------------------
# 특징 변환
# ---------------------------------------------------------------------------
def quad_features(P: np.ndarray) -> np.ndarray:
    """P:(n,3) = [Zone1, Zone2, Zone3] -> (n,9) 2차 특징."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    z1, z2, z3 = P[:, 0], P[:, 1], P[:, 2]
    return np.column_stack([z1, z2, z3,
                            z1 ** 2, z2 ** 2, z3 ** 2,
                            z1 * z2, z1 * z3, z2 * z3])


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def load_csv(path: str | Path):
    """학습 CSV -> (radius(R,), mrr(N,R), pressure(N,3), wafer_id(N,))."""
    path = Path(path)
    rows = [ln.split(",") for ln in
            path.read_text(encoding="utf-8-sig").strip().splitlines()]
    table = {r[0].strip(): r[1:] for r in rows[1:]}

    meta = {"Average", "Zone 1", "Zone 2", "Zone 3", "R-ring_real"}
    radius, prof = [], []
    for r in rows[1:]:
        key = r[0].strip()
        if key in meta or not key:
            continue
        radius.append(float(key))
        prof.append([float(v) for v in r[1:] if v.strip() != ""])

    radius = np.array(radius)
    mrr = np.array(prof).T                                     # (N,R)
    pressure = np.array([[float(v) for v in table[f"Zone {z}"] if v.strip() != ""]
                         for z in (1, 2, 3)]).T                # (N,3)
    wafer_id = np.array([int(float(c)) for c in rows[0][1:] if c.strip() != ""])
    return radius, mrr, pressure, wafer_id


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class QuadRidgeMRR:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    # -- 학습 -------------------------------------------------------------
    def fit(self, pressure: np.ndarray, mrr: np.ndarray, radius: np.ndarray):
        X = quad_features(pressure)
        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0) + 1e-12
        Z = np.column_stack([np.ones(len(X)), (X - self.x_mean_) / self.x_std_])

        G = np.eye(Z.shape[1]) * self.alpha
        G[0, 0] = 0.0                                   # 절편은 규제하지 않는다
        self.coef_ = np.linalg.solve(Z.T @ Z + G, Z.T @ mrr)     # (1+9, R)

        self.radius_ = np.asarray(radius, dtype=float)
        self.train_pressure_ = np.asarray(pressure, dtype=float)
        return self

    # -- 예측 -------------------------------------------------------------
    def predict(self, z1, z2, z3) -> np.ndarray:
        P = np.column_stack(np.broadcast_arrays(
            np.asarray(z1, float), np.asarray(z2, float), np.asarray(z3, float)))
        X = quad_features(P)
        Z = np.column_stack([np.ones(len(X)), (X - self.x_mean_) / self.x_std_])
        return Z @ self.coef_

    def predict_one(self, z1, z2, z3) -> np.ndarray:
        return self.predict(z1, z2, z3)[0]

    # -- 학습 압력 범위에서 얼마나 벗어났는가 --------------------------------
    def extrapolation_distance(self, z1, z2, z3) -> float:
        q = np.array([float(z1), float(z2), float(z3)])
        return float(np.min(np.linalg.norm(self.train_pressure_ - q, axis=1)))

    # -- 저장 / 로드 -------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": "QuadRidgeMRR",
            "alpha": self.alpha,
            "feature_names": FEATURE_NAMES,
            "radius": self.radius_.tolist(),
            "x_mean": self.x_mean_.tolist(),
            "x_std": self.x_std_.tolist(),
            "coef": self.coef_.tolist(),
            "train_pressure": self.train_pressure_.tolist(),
            "sigma": getattr(self, "sigma_", np.zeros(len(self.radius_))).tolist(),
            "cv": getattr(self, "cv_", {}),
        }, indent=1), encoding="utf-8")

    @staticmethod
    def load(path) -> "QuadRidgeMRR":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = QuadRidgeMRR(d["alpha"])
        m.radius_ = np.array(d["radius"])
        m.x_mean_ = np.array(d["x_mean"])
        m.x_std_ = np.array(d["x_std"])
        m.coef_ = np.array(d["coef"])
        m.train_pressure_ = np.array(d["train_pressure"])
        m.sigma_ = np.array(d["sigma"])
        m.cv_ = d.get("cv", {})
        return m


# ---------------------------------------------------------------------------
# WIWNU
# ---------------------------------------------------------------------------
def wiwnu(profile: np.ndarray, radius: np.ndarray) -> dict:
    """구간별 WIWNU [%] = 표본표준편차 / 평균 * 100."""
    profile = np.asarray(profile, dtype=float)
    radius = np.asarray(radius, dtype=float)
    out = {}
    for name, rmax in WIWNU_BANDS.items():
        m = np.abs(radius) <= rmax + 1e-9
        y = profile[m]
        out[name] = {
            "wiwnu_pct": float(y.std(ddof=1) / y.mean() * 100),
            "mean": float(y.mean()),
            "std": float(y.std(ddof=1)),
            "min": float(y.min()),
            "max": float(y.max()),
            "n_points": int(m.sum()),
            "range_mm": f"-{rmax:g} ~ +{rmax:g}",
        }
    return out


# ---------------------------------------------------------------------------
# 교차검증 (동일 압력 조건 웨이퍼를 통째로 빼는 Leave-One-Design-Out)
# ---------------------------------------------------------------------------
def design_groups(pressure: np.ndarray) -> np.ndarray:
    _, idx = np.unique(np.asarray(pressure), axis=0, return_inverse=True)
    return idx


def cross_validate(pressure, mrr, radius, alpha):
    g = design_groups(pressure)
    pred = np.zeros_like(mrr)
    for gi in np.unique(g):
        te = g == gi
        tr = ~te
        m = QuadRidgeMRR(alpha).fit(pressure[tr], mrr[tr], radius)
        pred[te] = m.predict(pressure[te, 0], pressure[te, 1], pressure[te, 2])
    return pred
