"""Train_1.csv (wide format) 로더.

CSV 레이아웃
-----------
row0            : "wafer No.", 1..21
row1..row43     : 반경(mm, +74 -> -74), 각 웨이퍼의 MRR
row "Average"   : 웨이퍼 평균 MRR
row "Zone 1/2/3": 존별 인가 압력 (psi)
row "R-ring_real": 리테이너 링 실제 압력 (psi)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ("Zone 1", "Zone 2", "Zone 3", "R-ring_real")


@dataclass
class CMPData:
    radius: np.ndarray      # (R,)  반경 [mm], +74 -> -74
    mrr: np.ndarray         # (N,R) 웨이퍼별 MRR 프로파일
    pressure: np.ndarray    # (N,4) Zone1,Zone2,Zone3,R-ring
    wafer_id: np.ndarray    # (N,)
    feature_names: tuple = FEATURE_NAMES

    @property
    def n_wafer(self) -> int:
        return self.mrr.shape[0]

    @property
    def n_radius(self) -> int:
        return self.mrr.shape[1]

    def design_groups(self) -> np.ndarray:
        """동일 압력 조건(=반복 실험) 웨이퍼에 같은 group id를 부여."""
        _, idx = np.unique(self.pressure[:, :3], axis=0, return_inverse=True)
        return idx


def load(csv_path: str | Path) -> CMPData:
    raw = pd.read_csv(csv_path, index_col=0)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    raw.index = raw.index.astype(str).str.strip()

    meta_rows = ["Average", "Zone 1", "Zone 2", "Zone 3", "R-ring_real"]
    prof = raw.drop(index=meta_rows)

    radius = prof.index.astype(float).to_numpy()
    mrr = prof.to_numpy(dtype=float).T                       # (N,R)
    pressure = raw.loc[list(FEATURE_NAMES)].to_numpy(dtype=float).T  # (N,4)
    wafer_id = np.array([int(float(c)) for c in raw.columns])

    if np.isnan(mrr).any() or np.isnan(pressure).any():
        raise ValueError("결측치가 포함되어 있습니다.")
    return CMPData(radius=radius, mrr=mrr, pressure=pressure, wafer_id=wafer_id)
