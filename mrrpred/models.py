"""MRR 프로파일 예측 모델 zoo.

공통 인터페이스
    fit(P, Y) -> self      P:(n,4) 압력, Y:(n,R) 프로파일
    predict(P) -> (n,R)
    params/set_params      그리드 서치용
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------------------------------------------------
# 특징 변환
# --------------------------------------------------------------------------
RING_FEATURES = {"zone+ring", "quad+ring", "phys"}


def uses_ring(kind: str) -> bool:
    """해당 특징셋이 리테이너 링 압력을 실제로 사용하는가."""
    return kind in RING_FEATURES


def make_features(P: np.ndarray, kind: str) -> np.ndarray:
    """P=(n,4): Zone1, Zone2, Zone3, R-ring."""
    z1, z2, z3, rg = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    if kind == "zone":                       # 존 압력만
        return np.c_[z1, z2, z3]
    if kind == "zone+ring":
        return np.c_[z1, z2, z3, rg]
    if kind == "quad":                       # 2차 + 교호작용
        return np.c_[z1, z2, z3, z1 ** 2, z2 ** 2, z3 ** 2,
                     z1 * z2, z1 * z3, z2 * z3]
    if kind == "quad+ring":
        return np.c_[z1, z2, z3, rg, z1 ** 2, z2 ** 2, z3 ** 2,
                     z1 * z2, z1 * z3, z2 * z3, rg * z1]
    if kind == "phys":
        # Preston 선형항 + 존간 압력차(=경계 전단/막 굽힘) + 링-에지 압력차
        return np.c_[z1, z2, z3, z1 - z2, z2 - z3, z1 - z3, rg - z1,
                     (z1 + z2 + z3) / 3]
    raise ValueError(kind)


class _Base:
    name = "base"

    def __init__(self, **kw):
        self.params = dict(kw)

    def set_params(self, **kw):
        self.params.update(kw)
        return self

    def __repr__(self):
        return f"{self.name}({self.params})"


# --------------------------------------------------------------------------
# 0. 기준선
# --------------------------------------------------------------------------
class MeanProfile(_Base):
    name = "MeanProfile"

    def fit(self, P, Y):
        self.mu_ = Y.mean(0)
        return self

    def predict(self, P):
        return np.tile(self.mu_, (len(P), 1))


class PrestonScale(_Base):
    """평균 형상 x (평균압력 선형 스케일) - 가장 단순한 물리 기준선."""
    name = "PrestonScale"

    def fit(self, P, Y):
        lvl = Y.mean(1)
        self.shape_ = (Y / lvl[:, None]).mean(0)
        X = np.c_[np.ones(len(P)), P[:, :3].mean(1)]
        self.c_, *_ = np.linalg.lstsq(X, lvl, rcond=None)
        return self

    def predict(self, P):
        lvl = np.c_[np.ones(len(P)), P[:, :3].mean(1)] @ self.c_
        return lvl[:, None] * self.shape_


# --------------------------------------------------------------------------
# 1. 반경별 능형회귀 (영향함수/중첩 모델)  +  반경방향 계수 평활
# --------------------------------------------------------------------------
class RadialRidge(_Base):
    """MRR(r) = a(r) + sum_z K_z(r)*P_z  를 반경마다 독립 적합.

    smooth>0 이면 계수장 K_z(r) 을 반경방향 가우시안 커널로 평활
    (물리적으로 영향함수는 r 에 대해 매끄럽다는 사전지식).
    log=True 이면 로그공간(=곱셈형 Preston)에서 적합.
    """
    name = "RadialRidge"

    def __init__(self, alpha=1.0, feat="zone", smooth=0.0, log=False):
        super().__init__(alpha=alpha, feat=feat, smooth=smooth, log=log)

    def fit(self, P, Y, radius=None):
        p = self.params
        self.radius_ = np.arange(Y.shape[1]) if radius is None else radius
        X = make_features(P, p["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        Xs = np.c_[np.ones(len(X)), (X - self.xm_) / self.xs_]
        T = np.log(Y) if p["log"] else Y

        G = np.eye(Xs.shape[1]) * p["alpha"]
        G[0, 0] = 0.0                                   # 절편은 규제 제외
        B = np.linalg.solve(Xs.T @ Xs + G, Xs.T @ T)     # (1+f, R)

        if p["smooth"] > 0:                             # 계수장 반경방향 평활
            d = self.radius_[:, None] - self.radius_[None, :]
            W = np.exp(-0.5 * (d / p["smooth"]) ** 2)
            W /= W.sum(1, keepdims=True)
            B = B @ W.T
        self.B_ = B
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        Xs = np.c_[np.ones(len(X)), (X - self.xm_) / self.xs_]
        out = Xs @ self.B_
        return np.exp(out) if self.params["log"] else out


# --------------------------------------------------------------------------
# 2. PCA 부분공간 + 회귀 (프로파일을 저차원으로 압축 후 예측)
# --------------------------------------------------------------------------
class PCARegressor(_Base):
    """Y ~= mu + sum_k score_k(P) * V_k ; score 를 지정 회귀기로 예측."""
    name = "PCA"

    def __init__(self, k=3, head="ridge", alpha=1.0, feat="zone", **kw):
        super().__init__(k=k, head=head, alpha=alpha, feat=feat, **kw)

    def _make_head(self):
        p = self.params
        h = p["head"]
        if h == "ridge":
            return None                                  # 직접 해석해
        if h == "gpr":
            ker = (ConstantKernel(1.0, (1e-3, 1e4))
                   * RBF(np.ones(self.X_.shape[1]), (1e-2, 1e3))
                   + WhiteKernel(1e-2, (1e-8, 1e1)))
            return GaussianProcessRegressor(kernel=ker, normalize_y=True,
                                            alpha=1e-10, n_restarts_optimizer=1,
                                            random_state=0)
        if h == "krr":
            return KernelRidge(alpha=p["alpha"], kernel="rbf",
                               gamma=p.get("gamma", 1.0))
        raise ValueError(h)

    def fit(self, P, Y, radius=None):
        p = self.params
        k = min(p["k"], min(Y.shape) - 1)
        self.mu_ = Y.mean(0)
        U, s, Vt = np.linalg.svd(Y - self.mu_, full_matrices=False)
        self.V_ = Vt[:k]                                  # (k,R)
        S = (Y - self.mu_) @ self.V_.T                    # (n,k) 점수

        X = make_features(P, p["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        self.X_ = (X - self.xm_) / self.xs_

        if p["head"] == "ridge":
            Xs = np.c_[np.ones(len(X)), self.X_]
            G = np.eye(Xs.shape[1]) * p["alpha"]
            G[0, 0] = 0.0
            self.B_ = np.linalg.solve(Xs.T @ Xs + G, Xs.T @ S)
            self.heads_ = None
        else:
            self.heads_ = []
            for j in range(k):
                h = self._make_head()
                h.fit(self.X_, S[:, j])
                self.heads_.append(h)
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        Xs = (X - self.xm_) / self.xs_
        if self.heads_ is None:
            S = np.c_[np.ones(len(X)), Xs] @ self.B_
        else:
            S = np.column_stack([h.predict(Xs) for h in self.heads_])
        return self.mu_ + S @ self.V_


# --------------------------------------------------------------------------
# 3. 범용 ML (전 반경 다출력)
# --------------------------------------------------------------------------
class SklearnMulti(_Base):
    name = "Sklearn"

    def __init__(self, kind="rf", feat="zone", **kw):
        super().__init__(kind=kind, feat=feat, **kw)
        self.name = f"ML[{kind}]"

    def _make(self):
        p, k = self.params, self.params["kind"]
        if k == "rf":
            return RandomForestRegressor(n_estimators=500, random_state=0,
                                         min_samples_leaf=p.get("leaf", 1))
        if k == "et":
            return ExtraTreesRegressor(n_estimators=500, random_state=0,
                                       min_samples_leaf=p.get("leaf", 1))
        if k == "gbr":
            return MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.1,
                random_state=0))
        if k == "krr":
            return KernelRidge(alpha=p.get("alpha", 1.0), kernel="rbf",
                               gamma=p.get("gamma", 1.0))
        if k == "gpr":
            ker = (ConstantKernel(1.0, (1e-3, 1e4))
                   * RBF(np.ones(self.nf_), (1e-2, 1e3))
                   + WhiteKernel(1e-2, (1e-8, 1e1)))
            return GaussianProcessRegressor(kernel=ker, normalize_y=True,
                                            alpha=1e-10, n_restarts_optimizer=1,
                                            random_state=0)
        if k == "pls":
            nc = max(1, min(p.get("n_comp", 2), self.nf_, self.n_ - 1))
            return PLSRegression(n_components=nc, scale=False)
        if k == "mlp":
            return MLPRegressor(hidden_layer_sizes=p.get("hidden", (64, 64)),
                                max_iter=4000, random_state=0,
                                alpha=p.get("alpha", 1.0), learning_rate_init=1e-2)
        raise ValueError(k)

    def fit(self, P, Y, radius=None):
        X = make_features(P, self.params["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        self.nf_, self.n_ = X.shape[1], X.shape[0]
        Xs = (X - self.xm_) / self.xs_
        self.ym_, self.ys_ = Y.mean(0), Y.std(0) + 1e-12
        self.m_ = self._make()
        self.m_.fit(Xs, (Y - self.ym_) / self.ys_)
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        Xs = (X - self.xm_) / self.xs_
        return np.asarray(self.m_.predict(Xs)).reshape(len(P), -1) * self.ys_ + self.ym_


# --------------------------------------------------------------------------
# 4. 대칭성 인지 모델 (물리 사전지식: 연마는 회전평균 -> 응답은 |r| 의 함수)
# --------------------------------------------------------------------------
class SymmetryRidge(_Base):
    """프로파일을 대칭/반대칭 성분으로 분해해 각각 다른 복잡도로 적합.

    Y(r) = Ys(|r|; P)  +  Ya(r; P)
      Ys : 압력 응답의 본체. |r| 로 접어 반경당 표본수 2배, 노이즈 1/sqrt(2).
      Ya : 헤드 틸트 등 장비 고유 비대칭. rank-q 저차원(고정 형상 x 압력 선형)으로
           강하게 규제 -> 자유파라미터를 21개 회귀에서 q개로 축소.
    """
    name = "SymmetryRidge"

    def __init__(self, alpha=1.0, feat="zone", rank=1, alpha_a=1.0, smooth=0.0):
        super().__init__(alpha=alpha, feat=feat, rank=rank,
                         alpha_a=alpha_a, smooth=smooth)

    @staticmethod
    def _ridge(X, T, alpha):
        Xs = np.c_[np.ones(len(X)), X]
        G = np.eye(Xs.shape[1]) * alpha
        G[0, 0] = 0.0
        return np.linalg.solve(Xs.T @ Xs + G, Xs.T @ T)

    def fit(self, P, Y, radius=None):
        p = self.params
        r = np.asarray(radius, dtype=float)
        self.r_ = r
        self.pos_ = np.where(r > 0)[0]
        self.zero_ = np.where(r == 0)[0]
        self.neg_ = np.array([int(np.where(r == -r[i])[0][0]) for i in self.pos_])

        Ys = (Y[:, self.pos_] + Y[:, self.neg_]) / 2          # (n, m)
        Ya = (Y[:, self.pos_] - Y[:, self.neg_]) / 2
        Y0 = Y[:, self.zero_]                                  # r=0 (있으면)

        X = make_features(P, p["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.xm_) / self.xs_

        # 대칭 성분: 접힌 반경마다 능형회귀
        self.Bs_ = self._ridge(Xs, np.c_[Ys, Y0], p["alpha"])
        if p["smooth"] > 0:
            rr = np.r_[np.abs(r[self.pos_]), np.abs(r[self.zero_])]
            d = rr[:, None] - rr[None, :]
            W = np.exp(-0.5 * (d / p["smooth"]) ** 2)
            W /= W.sum(1, keepdims=True)
            self.Bs_ = self.Bs_ @ W.T

        # 반대칭 성분: rank-q 고정 형상 + 압력 선형 진폭
        q = int(p["rank"])
        if q <= 0:
            self.A_, self.Ba_ = None, None
        else:
            U, s, Vt = np.linalg.svd(Ya, full_matrices=False)
            q = min(q, len(s))
            self.A_ = Vt[:q]                                   # (q,m) 고정 형상
            self.Ba_ = self._ridge(Xs, Ya @ self.A_.T, p["alpha_a"])
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        Xs = np.c_[np.ones(len(X)), (X - self.xm_) / self.xs_]
        S = Xs @ self.Bs_
        m = len(self.pos_)
        Ys, Y0 = S[:, :m], S[:, m:]
        Ya = (Xs @ self.Ba_) @ self.A_ if self.A_ is not None else np.zeros_like(Ys)

        out = np.zeros((len(P), len(self.r_)))
        out[:, self.pos_] = Ys + Ya
        out[:, self.neg_] = Ys - Ya
        out[:, self.zero_] = Y0
        return out


# --------------------------------------------------------------------------
# 5. 선형 물리모델 + 잔차 GP (비선형 보정 & 불확실도)
# --------------------------------------------------------------------------
class RidgePlusGP(_Base):
    """1단: 반경별 선형 영향함수. 2단: 잔차를 PCA 저차원에서 GP 로 보정."""
    name = "Ridge+GP"

    def __init__(self, alpha=1.0, feat="zone", rank=1, k=2, smooth=0.0):
        super().__init__(alpha=alpha, feat=feat, rank=rank, k=k, smooth=smooth)

    def fit(self, P, Y, radius=None):
        p = self.params
        self.base_ = SymmetryRidge(alpha=p["alpha"], feat=p["feat"],
                                   rank=p["rank"], alpha_a=p["alpha"],
                                   smooth=p["smooth"]).fit(P, Y, radius=radius)
        Rres = Y - self.base_.predict(P)
        k = min(p["k"], min(Rres.shape) - 1)
        U, s, Vt = np.linalg.svd(Rres - Rres.mean(0), full_matrices=False)
        self.mu_ = Rres.mean(0)
        self.V_ = Vt[:k]
        S = (Rres - self.mu_) @ self.V_.T

        X = make_features(P, p["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.xm_) / self.xs_
        self.heads_ = []
        for j in range(k):
            ker = (ConstantKernel(1.0, (1e-3, 1e4)) * RBF(np.ones(Xs.shape[1]), (1e-1, 1e3))
                   + WhiteKernel(1e-1, (1e-6, 1e2)))
            g = GaussianProcessRegressor(kernel=ker, normalize_y=True, alpha=1e-10,
                                         n_restarts_optimizer=1, random_state=0)
            g.fit(Xs, S[:, j])
            self.heads_.append(g)
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        Xs = (X - self.xm_) / self.xs_
        S = np.column_stack([h.predict(Xs) for h in self.heads_])
        return self.base_.predict(P) + self.mu_ + S @ self.V_


# --------------------------------------------------------------------------
# 6. 저계수(reduced-rank) 회귀 + 대칭성 분해
# --------------------------------------------------------------------------
class ReducedRankRidge(_Base):
    """능형회귀 계수행렬을 rank-q 로 절단. 43개 반경 회귀가 q개 잠재모드를 공유."""
    name = "ReducedRankRidge"

    def __init__(self, alpha=1.0, feat="zone", rank=2):
        super().__init__(alpha=alpha, feat=feat, rank=rank)

    def fit(self, P, Y, radius=None):
        p = self.params
        X = make_features(P, p["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.xm_) / self.xs_
        self.ym_ = Y.mean(0)
        G = np.eye(Xs.shape[1]) * p["alpha"]
        B = np.linalg.solve(Xs.T @ Xs + G, Xs.T @ (Y - self.ym_))
        F = Xs @ B                                   # 적합값 (n,R)
        q = min(int(p["rank"]), min(F.shape))
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        Pq = Vt[:q].T @ Vt[:q]                       # rank-q 사영
        self.B_ = B @ Pq
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        return self.ym_ + ((X - self.xm_) / self.xs_) @ self.B_


class SymmetryPLS(_Base):
    """대칭/반대칭 분해 + 대칭 성분을 PLS(지도학습 저차원)로 적합."""
    name = "SymmetryPLS"

    def __init__(self, n_comp=2, feat="zone", rank=1, alpha_a=1.0):
        super().__init__(n_comp=n_comp, feat=feat, rank=rank, alpha_a=alpha_a)

    def fit(self, P, Y, radius=None):
        p = self.params
        r = np.asarray(radius, dtype=float)
        self.r_ = r
        self.pos_ = np.where(r > 0)[0]
        self.zero_ = np.where(r == 0)[0]
        self.neg_ = np.array([int(np.where(r == -r[i])[0][0]) for i in self.pos_])

        Ys = np.c_[(Y[:, self.pos_] + Y[:, self.neg_]) / 2, Y[:, self.zero_]]
        Ya = (Y[:, self.pos_] - Y[:, self.neg_]) / 2

        X = make_features(P, p["feat"])
        self.xm_, self.xs_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.xm_) / self.xs_

        nc = max(1, min(int(p["n_comp"]), Xs.shape[1], len(Xs) - 1))
        self.pls_ = PLSRegression(n_components=nc, scale=False).fit(Xs, Ys)

        q = int(p["rank"])
        if q <= 0:
            self.A_, self.Ba_ = None, None
        else:
            U, s, Vt = np.linalg.svd(Ya, full_matrices=False)
            q = min(q, len(s))
            self.A_ = Vt[:q]
            Z = np.c_[np.ones(len(Xs)), Xs]
            G = np.eye(Z.shape[1]) * p["alpha_a"]
            G[0, 0] = 0.0
            self.Ba_ = np.linalg.solve(Z.T @ Z + G, Z.T @ (Ya @ self.A_.T))
        return self

    def predict(self, P):
        X = make_features(P, self.params["feat"])
        Xs = (X - self.xm_) / self.xs_
        S = np.asarray(self.pls_.predict(Xs)).reshape(len(P), -1)
        m = len(self.pos_)
        Ys, Y0 = S[:, :m], S[:, m:]
        if self.A_ is not None:
            Ya = (np.c_[np.ones(len(Xs)), Xs] @ self.Ba_) @ self.A_
        else:
            Ya = np.zeros_like(Ys)
        out = np.zeros((len(P), len(self.r_)))
        out[:, self.pos_] = Ys + Ya
        out[:, self.neg_] = Ys - Ya
        out[:, self.zero_] = Y0
        return out
