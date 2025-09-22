import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

# ---- import your core class ----
import os, sys
try:
    from SNMF.nmf_cpu import NMF as CoreNMF
except ModuleNotFoundError:
    # add repo root: .../GitHub/SNMF
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, REPO_ROOT)
    from SNMF.nmf_cpu import NMF as CoreNMF


# from SNMF.nmf_cpu import NMF as CoreNMF


# ---- small safety patch: stable torch softmax for Y_hat ----
def _softmax_torch(Z):
    Z = Z - torch.amax(Z, dim=1, keepdim=True)  # (1,C,N) max-shift
    e_x = torch.exp(Z)
    return e_x / torch.sum(e_x, dim=1, keepdim=True).clamp_min(1e-30)

def _patch_core():
    # 1) Make Y_hat pure-torch
    def Y_hat(self):
        Z = self.B @ self.H  # (1,C,K)@(1,K,N)->(1,C,N)
        return _softmax_torch(Z)
    CoreNMF.Y_hat = property(Y_hat)

    # 2) Respect init_method (your current _initialise_wh overrides to 'random')
    orig_init = CoreNMF._initialise_wh
    def _initialise_wh(self, init_method):
        return orig_init(self, init_method=getattr(self, "_init_method", "random"))
    CoreNMF._initialise_wh = _initialise_wh

    # 3) Record init_method in __init__
    orig_ctor = CoreNMF.__init__
    def __init__(self, *args, init_method='random', **kwargs):
        self._init_method = init_method
        return orig_ctor(self, *args, init_method=init_method, **kwargs)
    CoreNMF.__init__ = __init__

_patch_core()


class SNMFClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Dataset-agnostic supervised NMF classifier wrapping your SNMF core.
    X: (n_samples, n_features) non-negative
    y: 1D labels or one-hot (n_samples, n_classes)
    """

    def __init__(
        self,
        n_components=10,
        lambda_c=0.1,
        lambda_p=0.0,
        lr=0.003,
        max_iterations=200000,
        min_iterations=2000,
        tolerance=1e-8,
        tolerance_ce=1e-2,
        test_conv=1000,
        report_loss=1000,
        init_method="random",              # 'random' works; enable NNDSVD later if desired
        floating_point_precision="double",
        random_state=42,
        beta=2,
        supervised=True,
        generator=None,
    ):
        self.n_components = n_components
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p
        self.lr = lr
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.tolerance = tolerance
        self.tolerance_ce = tolerance_ce
        self.test_conv = test_conv
        self.report_loss = report_loss
        self.init_method = init_method
        self.floating_point_precision = floating_point_precision
        self.random_state = random_state
        self.beta = beta
        self.supervised = supervised
        self.generator = generator

    # ----- utilities -----
    def _check_Xy(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if np.any(X < 0):
            raise ValueError("SNMF requires non-negative X.")

        y = np.asarray(y)
        if y.ndim == 1:
            self._ohe = OneHotEncoder(sparse_output=False, dtype=np.float64)
            Y = self._ohe.fit_transform(y.reshape(-1, 1))
            self.classes_ = self._ohe.categories_[0]
        elif y.ndim == 2 and set(np.unique(y)).issubset({0, 1}) and y.shape[0] == X.shape[0]:
            self._ohe = None
            Y = y.astype(np.float64)
            self.classes_ = np.arange(Y.shape[1])
        else:
            raise ValueError("y must be 1D labels or one-hot with shape (n_samples, n_classes).")
        return X, Y

    @staticmethod
    def _to_core_shapes(X, Y):
        # Core expects (1, F, N) and (1, C, N)
        V = X.T[None, :, :]       # (1, features, samples)
        Yc = Y.T[None, :, :]      # (1, classes, samples)
        return V, Yc

    # ----- sklearn API -----
    def fit(self, X, y):
        rng = self.generator or np.random.default_rng(self.random_state)

        X, Y = self._check_Xy(X, y)
        V, Yc = self._to_core_shapes(X, Y)

        # Instantiate your core
        self.core_ = CoreNMF(
            V=torch.from_numpy(V),
            Y=torch.from_numpy(Yc),
            rank=self.n_components,
            lambda_c=self.lambda_c,
            lambda_p=self.lambda_p,
            lr=self.lr,
            report_loss=self.report_loss,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            tolerance_ce=self.tolerance_ce,
            test_conv=self.test_conv,
            init_method=self.init_method,
            floating_point_precision=self.floating_point_precision,
            min_iterations=self.min_iterations,
            generator=rng,
        )
        self.core_.fit(beta=self.beta, supervised=self.supervised)

        # Extract learned params; remove batch dim
        self.W_ = self.core_.W.detach().cpu().numpy()[0]   # (F, K)
        self.H_ = self.core_.H.detach().cpu().numpy()[0]   # (K, N)
        self.B_ = self.core_.B.detach().cpu().numpy()[0]   # (C, K)

        self.reconstruction_err_ = float(self.core_._fro_loss.item())
        self.history_ = dict(self.core_.results)
        self.converged_at_ = int(getattr(self.core_, "_conv", 0))
        return self

    def transform(self, X):
        """Project new X -> H with W fixed (uses your core.refit, lambda_c=0 inside)."""
        check_is_fitted(self, "W_")
        X = np.asarray(X, dtype=np.float64)
        if np.any(X < 0):
            raise ValueError("SNMF requires non-negative X.")

        # Dummy Y (not used because lambda_c=0 in refit)
        C = len(self.classes_)
        Y_dummy = np.zeros((X.shape[0], C), dtype=np.float64)
        V, Yc = self._to_core_shapes(X, Y_dummy)

        self.core_._V = torch.from_numpy(V).type(self.core_._tensor_type)
        self.core_._Y = torch.from_numpy(Yc).type(self.core_._tensor_type)

        # Fix W and update H (and B per your refit implementation)
        self.core_.refit(self.W_, beta=self.beta, supervised=False)
        H_new = self.core_.H.detach().cpu().numpy()[0]   # (K, N_new)
        return H_new.T                                    # (n_samples_new, K)

    def decision_function(self, X):
        check_is_fitted(self, "B_")
        H = self.transform(X)              # (n_samples, K)
        Z = self.B_ @ H.T                  # (C, n_samples)
        return Z.T

    def predict_proba(self, X):
        Z = self.decision_function(X)
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True).clip(min=1e-30)
        return P

    def predict(self, X):
        idx = self.predict_proba(X).argmax(axis=1)
        return np.asarray(self.classes_)[idx]

    def reconstruction_error(self, X):
        check_is_fitted(self, "W_")
        H = self.transform(X).T               # (K, N)
        X_hat = (self.W_ @ H).T               # (N, F)
        num = np.linalg.norm(X - X_hat, ord='fro')
        den = np.linalg.norm(X, ord='fro') + 1e-30
        return float(num / den)
