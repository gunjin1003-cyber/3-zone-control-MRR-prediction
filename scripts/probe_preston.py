"""배포 모델이 Preston 법칙(MRR = k·P·V)을 따르는 모델인가?

Preston 법칙의 요구조건 3가지를 하나씩 검증한다.
  (1) 압력에 대해 1차   (지수 n = 1)
  (2) 원점을 지남       (절편 = 0)
  (3) 국소적으로 성립   (MRR(r) = k(r)·P_contact(r))
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from mrrpred.data import load
from mrrpred import models as M
from mrrpred.cv import folds, rmse

d = load("data/Train_1.csv")
P, Y, r, g = d.pressure, d.mrr, d.radius, d.design_groups()
ctr, edge = np.abs(r) <= 50, np.abs(r) >= 70


def cv_generic(fit_fn):
    pred = np.zeros_like(Y)
    for tr, te in folds(g):
        pred[te] = fit_fn(P[tr], Y[tr])(P[te])
    return pred


def ridge_fn(feat, alpha, intercept=True):
    """절편 유무만 다르게 하고 규제 조건은 동일하게 맞춘다.

    두 경우 모두 특징을 표준편차로만 나눠 스케일을 맞추고(중심화는 하지 않음 —
    중심화 자체가 절편을 암묵적으로 넣는 것과 같다), 기울기 계수에만 alpha 를 건다.
    """
    def fit(Ptr, Ytr):
        X = M.make_features(Ptr, feat)
        xs = X.std(0) + 1e-12
        Xs = X / xs
        if intercept:
            Z = np.c_[np.ones(len(Xs)), Xs]
            G = np.eye(Z.shape[1]) * alpha; G[0, 0] = 0.0
        else:
            Z = Xs
            G = np.eye(Z.shape[1]) * alpha
        B = np.linalg.solve(Z.T @ Z + G, Z.T @ Ytr)

        def pred(Pte):
            Xe = M.make_features(Pte, feat) / xs
            Ze = np.c_[np.ones(len(Xe)), Xe] if intercept else Xe
            return Ze @ B
        return pred
    return fit


ALPHAS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)


def best_over_alpha(feat, intercept):
    """각 형태마다 alpha 를 최적으로 잡아 공정하게 비교."""
    out = None
    for al in ALPHAS:
        p = cv_generic(ridge_fn(feat, al, intercept))
        if out is None or rmse(Y, p) < out[0]:
            out = (rmse(Y, p), al, p)
    return out


print("=" * 74)
print("(1)+(2) Preston 형태를 하나씩 풀어주며 CV 오차 변화")
print(f"{'모델':44s}{'전체':>8s}{'중심':>8s}{'에지':>9s}")
print("-" * 74)
rows = [
    ("엄격 Preston  : MRR(r)=ΣK_z(r)·P_z (절편없음,1차)", "zone", False),
    ("  + 절편      : a(r)+ΣK_z(r)·P_z",                  "zone", True),
    ("  + 2차/교호  : 배포 모델의 특징셋",                  "quad", True),
]
for nm, feat, ic in rows:
    sc, al, p = best_over_alpha(feat, ic)
    print(f"{nm:40s}{sc:8.1f}{rmse(Y[:, ctr], p[:, ctr]):8.1f}"
          f"{rmse(Y[:, edge], p[:, edge]):9.1f}   (α={al:g})")
print("-" * 74)
print(f"{'전역 형상 Preston (평균형상 x 압력선형)':40s}", end="")
pp = np.zeros_like(Y)
for tr, te in folds(g):
    m = M.PrestonScale().fit(P[tr], Y[tr]); pp[te] = m.predict(P[te])
print(f"{rmse(Y, pp):8.1f}{rmse(Y[:, ctr], pp[:, ctr]):8.1f}{rmse(Y[:, edge], pp[:, edge]):9.1f}")

print("=" * 74)
print("(2) 배포 모델의 절편 a(r) — Preston 이면 0 이어야 한다")
m = M.RadialRidge(alpha=1.0, feat="zone").fit(P, Y, radius=r)
zero = m.predict(np.zeros((1, 4)))[0]           # P=0 에서의 예측
print(f"   P=0 예측  중심 평균 {zero[ctr].mean():+8.1f} A/min   "
      f"(실측 중심 평균 {Y[:, ctr].mean():.0f} 대비 {zero[ctr].mean()/Y[:, ctr].mean()*100:+.1f}%)")
print(f"             에지 평균 {zero[edge].mean():+8.1f} A/min   "
      f"(실측 에지 평균 {Y[:, edge].mean():.0f} 대비 {zero[edge].mean()/Y[:, edge].mean()*100:+.1f}%)")

print("=" * 74)
print("(3) 국소 Preston: 학습된 영향함수는 존별 압력의 국소 가중합인가")
B = m.B_[1:] / m.xs_[:, None]                   # [A/min per psi], (3,R)
tot = B.sum(0)
print("   k(r) = Σ_z dMRR/dP_z = 세 존을 동시에 1 psi 올릴 때의 MRR 증가.")
print("   Preston 이면 k(r) = k_p·V 로 r 에 무관하게 일정해야 한다.")
print("   (가중치는 k(r) 로 나눈 값이라 합이 1 인 것은 정의상 당연 — 볼 것은 '어느 존에")
print("    몰려 있는가' 와 'k(r) 이 평평한가' 두 가지다.)")
for i in np.argsort(-r)[::4]:
    w = B[:, i] / tot[i]
    print(f"   r={r[i]:+4.0f}  k(r)={tot[i]:7.1f}   가중치 [Z1 {w[0]:+.2f}  Z2 {w[1]:+.2f}  Z3 {w[2]:+.2f}]")
