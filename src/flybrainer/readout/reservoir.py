"""Readout 2: the reservoir (liquid state machine) readout.

Feature vector: log(1 + spike count) for every neuron of the selected
populations (default DN, MBON, random2000), standardized with the training
mean and std. A closed-form ridge regression predicts realized-over-implied
movement over the configured horizon; a logistic classifier predicts whether
it exceeds one. Policy with hysteresis: ENTER when the prediction is below
1 - tau and flat, EXIT when above 1 + tau while in a position, else HOLD.

An untrained readout still predicts (HOLD, confidence 0) so the pipeline runs
before any fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from flybrainer.interfaces import Decision, Prediction, SensorFrame
from flybrainer.readout.fixed import ColumnBinding

DEFAULT_POPULATIONS = ("DN", "MBON", "random2000")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class ReservoirReadout:
    name = "reservoir"

    def __init__(
        self,
        populations=DEFAULT_POPULATIONS,
        alpha: float = 10.0,
        tau: float = 0.1,
        horizon_minutes: int = 60,
        neural_ms: float = 200.0,
        brain=None,
        classifier: bool = True,
        std_floor: float = 1e-6,
    ):
        self.populations = tuple(populations)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.horizon_minutes = int(horizon_minutes)
        self.neural_ms = float(neural_ms)
        self.use_classifier = bool(classifier)
        self.std_floor = float(std_floor)

        self.neuron_index: np.ndarray | None = None  # neuron ids, one per feature
        self.feature_population: np.ndarray | None = None  # index into self.populations per feature
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: float = 1.0
        self.clf_coef: np.ndarray | None = None
        self.clf_intercept: float = 0.0
        self.fitted = False
        self.fit_info: dict = {}

        self._binding: ColumnBinding | None = None
        self._positions: np.ndarray | None = None
        self._eig: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if brain is not None:
            self.resolve(brain)

    # -- configuration -------------------------------------------------------

    def params(self) -> dict:
        return {
            "name": self.name,
            "version": 1,
            "populations": list(self.populations),
            "alpha": self.alpha,
            "tau": self.tau,
            "horizon_minutes": self.horizon_minutes,
            "neural_ms": self.neural_ms,
            "classifier": self.use_classifier,
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.params(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def n_features(self) -> int:
        return 0 if self.neuron_index is None else int(len(self.neuron_index))

    # -- feature selection ---------------------------------------------------

    def resolve(self, brain) -> np.ndarray:
        """Resolve the feature neuron ids from the brain's populations (sorted, unique)."""
        ids = []
        owner = []
        for k, name in enumerate(self.populations):
            if name not in brain.populations:
                raise KeyError(f"brain has no population {name!r}")
            pop = np.asarray(brain.populations[name], dtype=np.int64)
            ids.append(pop)
            owner.append(np.full(len(pop), k, dtype=np.int64))
        all_ids = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64)
        all_owner = np.concatenate(owner) if owner else np.zeros(0, dtype=np.int64)
        uniq, first = np.unique(all_ids, return_index=True)
        self.set_neuron_index(uniq, all_owner[first])
        return uniq

    def set_neuron_index(self, neuron_ids: np.ndarray, feature_population: np.ndarray | None = None) -> None:
        self.neuron_index = np.ascontiguousarray(neuron_ids, dtype=np.int64)
        if feature_population is None:
            feature_population = np.zeros(len(self.neuron_index), dtype=np.int64)
        self.feature_population = np.ascontiguousarray(feature_population, dtype=np.int64)
        self._positions = None

    def bind_columns(self, neuron_ids: np.ndarray) -> None:
        """Declare that count matrices passed later hold these neuron ids as columns."""
        self._binding = ColumnBinding(neuron_ids)
        self._positions = None

    def _select(self, counts: np.ndarray, brain=None) -> np.ndarray:
        if self.neuron_index is None:
            if brain is None:
                raise ValueError("readout has no neuron index; pass a brain or call resolve")
            self.resolve(brain)
        counts = np.asarray(counts)
        n_last = counts.shape[-1]
        if n_last == self.n_features and (self._binding is None or brain is None or n_last != brain.n):
            return counts
        if brain is not None and n_last == brain.n:
            return counts[..., self.neuron_index]
        if self._binding is not None:
            if self._positions is None:
                self._positions = self._binding.positions(self.neuron_index)
            return counts[..., self._positions]
        raise ValueError(
            f"cannot map counts of width {n_last} to {self.n_features} features; bind columns or pass a brain"
        )

    def features(self, counts: np.ndarray, brain=None) -> np.ndarray:
        return np.log1p(np.asarray(self._select(counts, brain), dtype=np.float64))

    def _standardize(self, F: np.ndarray) -> np.ndarray:
        return (F - self.mean) / self.std

    # -- fitting -------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray | None = None, classifier: bool | None = None):
        """Fit ridge (and the logistic classifier) on counts X (n_obs, n_cols) and targets y.

        X holds raw spike counts. If `columns` gives the neuron ids of X's
        columns, the readout selects its own features from them; otherwise X
        must already be (n_obs, n_features) in neuron_index order.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        if columns is not None:
            self.bind_columns(np.asarray(columns))
            if self.neuron_index is None:
                self.set_neuron_index(np.asarray(columns))
        elif self.neuron_index is None:
            self.set_neuron_index(np.arange(X.shape[1]))
        F = self.features(X)
        keep = np.isfinite(y)
        F = F[keep]
        y = y[keep]
        if F.shape[0] < 2:
            raise ValueError("need at least two observations to fit")
        self.mean = F.mean(axis=0)
        std = F.std(axis=0)
        std = np.where(std < self.std_floor, 1.0, std)
        self.std = std
        Z = self._standardize(F).astype(np.float32)
        gram = (Z.T @ Z).astype(np.float64)
        ymean = float(y.mean())
        zty = (Z.T @ (y - ymean).astype(np.float32)).astype(np.float64)
        evals, evecs = np.linalg.eigh(gram)
        evals = np.clip(evals, 0.0, None)
        self._eig = (evals, evecs, zty)
        self.intercept = ymean
        self.refit_alpha(self.alpha)
        self.fitted = True
        self.fit_info = {"n_obs": int(F.shape[0]), "n_features": int(F.shape[1]), "y_mean": ymean}
        use_clf = self.use_classifier if classifier is None else bool(classifier)
        if use_clf:
            self.fit_classifier(Z, y > 1.0)
        else:
            self.clf_coef = None
        return self

    def refit_alpha(self, alpha: float) -> None:
        """Recompute the ridge weights for a new alpha from the stored decomposition."""
        if self._eig is None:
            raise ValueError("fit() must run before refit_alpha")
        evals, evecs, zty = self._eig
        self.alpha = float(alpha)
        proj = evecs.T @ zty
        self.coef = evecs @ (proj / (evals + self.alpha))

    def fit_classifier(self, Z: np.ndarray, labels: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=bool)
        if labels.all() or (~labels).all():
            self.clf_coef = None
            self.clf_intercept = 0.0
            return
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:  # pragma: no cover
            LogisticRegression = None  # type: ignore  # pragma: no cover
            self.clf_coef = None
            return
        C = 1.0 / (2.0 * max(self.alpha, 1e-6))
        clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
        clf.fit(Z, labels.astype(int))
        self.clf_coef = clf.coef_.ravel().astype(np.float64)
        self.clf_intercept = float(clf.intercept_[0])

    def fit_classifier_from_counts(self, X: np.ndarray, y: np.ndarray) -> None:
        F = self.features(X)
        keep = np.isfinite(y)
        Z = self._standardize(F[keep])
        self.fit_classifier(Z, np.asarray(y)[keep] > 1.0)

    # -- prediction ----------------------------------------------------------

    def signal(self, prediction: float) -> Decision:
        """Position-free signal from the hysteresis band."""
        if prediction < 1.0 - self.tau:
            return Decision.ENTER
        if prediction > 1.0 + self.tau:
            return Decision.EXIT
        return Decision.HOLD

    def policy(self, prediction: float, in_position: bool) -> Decision:
        s = self.signal(prediction)
        if s == Decision.ENTER and not in_position:
            return Decision.ENTER
        if s == Decision.EXIT and in_position:
            return Decision.EXIT
        return Decision.HOLD

    def predict_batch(self, X: np.ndarray, brain=None) -> dict[str, np.ndarray]:
        """Ridge predictions and classifier probabilities for rows of counts."""
        X = np.asarray(X)
        n = X.shape[0]
        if not self.fitted:
            return {"roi": np.ones(n), "prob_above_1": np.full(n, 0.5), "confidence": np.zeros(n)}
        Z = self._standardize(self.features(X, brain))
        roi = Z @ self.coef + self.intercept
        if self.clf_coef is not None:
            prob = _sigmoid(Z @ self.clf_coef + self.clf_intercept)
            conf = np.abs(2.0 * prob - 1.0)
        else:
            prob = (roi > 1.0).astype(np.float64)
            conf = np.clip(np.abs(roi - 1.0) / (2.0 * max(self.tau, 1e-6)), 0.0, 1.0)
        return {"roi": roi, "prob_above_1": prob, "confidence": conf}

    def top_populations(self, z: np.ndarray, k: int = 5) -> list[list]:
        if self.coef is None or self.feature_population is None:
            return []
        contrib = self.coef * z
        out = []
        for i, name in enumerate(self.populations):
            mask = self.feature_population == i
            if mask.any():
                out.append([name, float(contrib[mask].sum())])
        out.sort(key=lambda item: abs(item[1]), reverse=True)
        return out[:k]

    def predict(self, counts: np.ndarray, brain=None, observation: SensorFrame | None = None) -> Prediction:
        in_position = bool(observation is not None and observation.position_lots != 0)
        if not self.fitted:
            return Prediction(
                realized_over_implied=1.0,
                confidence=0.0,
                decision=Decision.HOLD,
                details={"fitted": False, "readout": self.name, "tau": self.tau},
            )
        z = self._standardize(self.features(np.asarray(counts).reshape(1, -1), brain))[0]
        roi = float(z @ self.coef + self.intercept)
        if self.clf_coef is not None:
            prob = float(_sigmoid(z @ self.clf_coef + self.clf_intercept))
            confidence = abs(2.0 * prob - 1.0)
        else:
            prob = 1.0 if roi > 1.0 else 0.0
            confidence = float(np.clip(abs(roi - 1.0) / (2.0 * max(self.tau, 1e-6)), 0.0, 1.0))
        decision = self.policy(roi, in_position)
        details = {
            "fitted": True,
            "readout": self.name,
            "tau": self.tau,
            "signal": self.signal(roi).value,
            "prob_above_1": prob,
            "ridge_alpha": self.alpha,
            "features": self.n_features,
            "top_populations": self.top_populations(z),
            "horizon_minutes": self.horizon_minutes,
        }
        if observation is not None and observation.straddle_premium:
            details["implied_move_points"] = float(observation.straddle_premium)
            details["predicted_move_points"] = float(roi * observation.straddle_premium)
        return Prediction(
            realized_over_implied=roi, confidence=float(confidence), decision=decision, details=details
        )

    # -- persistence ---------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = dict(self.params())
        meta.update({"fitted": self.fitted, "n_features": self.n_features, "fit_info": self.fit_info})
        (directory / "readout.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        arrays = {
            "neuron_index": self.neuron_index if self.neuron_index is not None else np.zeros(0, np.int64),
            "feature_population": (
                self.feature_population if self.feature_population is not None else np.zeros(0, np.int64)
            ),
        }
        if self.fitted:
            arrays.update(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "coef": self.coef,
                    "intercept": np.array([self.intercept]),
                    "clf_coef": self.clf_coef if self.clf_coef is not None else np.zeros(0),
                    "clf_intercept": np.array([self.clf_intercept]),
                }
            )
        np.savez_compressed(directory / "readout.npz", **arrays)

    @classmethod
    def load(cls, directory: str | Path, brain=None) -> ReservoirReadout:
        directory = Path(directory)
        meta = json.loads((directory / "readout.json").read_text(encoding="utf-8"))
        obj = cls(
            populations=tuple(meta.get("populations", DEFAULT_POPULATIONS)),
            alpha=float(meta.get("alpha", 10.0)),
            tau=float(meta.get("tau", 0.1)),
            horizon_minutes=int(meta.get("horizon_minutes", 60)),
            neural_ms=float(meta.get("neural_ms", 200.0)),
            classifier=bool(meta.get("classifier", True)),
        )
        with np.load(directory / "readout.npz") as data:
            if data["neuron_index"].size:
                obj.set_neuron_index(data["neuron_index"], data["feature_population"])
            if meta.get("fitted"):
                obj.mean = data["mean"]
                obj.std = data["std"]
                obj.coef = data["coef"]
                obj.intercept = float(data["intercept"][0])
                obj.clf_coef = data["clf_coef"] if data["clf_coef"].size else None
                obj.clf_intercept = float(data["clf_intercept"][0])
                obj.fitted = True
                obj.fit_info = meta.get("fit_info", {})
        if obj.neuron_index is None and brain is not None:
            obj.resolve(brain)
        return obj
