"""최종 모델 학습 -> models/mrr_model.pkl 저장."""
import sys, pathlib, argparse, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.pipeline import MRRPredictor

# benchmark.py 결과로 확정된 기본 설정 (reports/best_config.json 로 덮어쓰기 가능)
DEFAULT = dict(model="SymmetryRidge",
               params=dict(alpha=1.0, feat="quad", rank=1, alpha_a=1.0, smooth=0.0))

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/Train_1.csv")
ap.add_argument("--out", default="models/mrr_model.pkl")
ap.add_argument("--config", default="reports/best_config.json")
a = ap.parse_args()

cfg = DEFAULT
if pathlib.Path(a.config).exists():
    cfg = json.load(open(a.config))
    print(f"[config] {a.config} 사용")

d = load(a.csv)
p = MRRPredictor(model_cls=getattr(M, cfg["model"]), params=cfg["params"])
p.fit(d)
p.save(a.out)

print(f"모델      : {cfg['model']}  {cfg['params']}")
print(f"학습 웨이퍼: {d.n_wafer}장 x {d.n_radius}점")
print(f"교차검증   : RMSE {p.cv_['rmse']:.1f} A/min,  MAPE {p.cv_['mape']:.2f}%")
if p.needs_ring:
    print(f"링압력 보간: ring = {p.ring_coef_[0]:.3f} + {p.ring_coef_[1]:.3f}*Z1 "
          f"+ {p.ring_coef_[2]:.3f}*Z2 + {p.ring_coef_[3]:.3f}*Z3   (RMSE {p.ring_rmse_:.3f} psi)")
else:
    print("링압력    : 사용 안 함 (존 압력 3개만 입력하면 됩니다)")
print(f"저장      : {a.out}")
