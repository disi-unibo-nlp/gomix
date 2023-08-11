from sklearn.linear_model import LinearRegression
import numpy as np
import math


class StackingMetaLearner:
    def __init__(self, n_classes: int):
        if n_classes < 1:
            raise ValueError('n_classes must be a positive integer.')

        self._n_classes = n_classes
        self._n_regressions = min(n_classes, 60)
        self._classes_per_regression = math.ceil(n_classes / self._n_regressions)
        self._regressions = [LinearRegression(fit_intercept=False) for _ in range(self._n_regressions)]

    def fit(self, base_scores: np.ndarray, labels: np.ndarray):
        """
        Base scores: (n_train_samples, n_classes, n_base_models)
        Labels: (n_train_samples, n_classes)
        """
        assert base_scores.ndim == 3 and labels.ndim == 2 \
               and base_scores.shape[0] == labels.shape[0] \
               and base_scores.shape[1] == self._n_classes \
               and labels.shape[1] == self._n_classes

        n_base_models = base_scores.shape[2]

        for idx in range(self._n_regressions):
            start_class = idx * self._classes_per_regression
            end_class = min((idx + 1) * self._classes_per_regression, self._n_classes)

            X_train = base_scores[:, start_class:end_class, :].reshape((-1, n_base_models))
            y_train = labels[:, start_class:end_class].reshape(-1, 1)

            self._regressions[idx].fit(X_train, y_train)

    def predict(self, base_scores: np.ndarray) -> np.ndarray:
        """
        Base scores: (n_samples, n_classes, n_base_models)
        """
        assert base_scores.ndim == 3 and base_scores.shape[1] == self._n_classes

        predictions = np.zeros((base_scores.shape[0], self._n_classes))
        n_base_models = base_scores.shape[2]

        for idx in range(self._n_regressions):
            start_class = idx * self._classes_per_regression
            end_class = min((idx + 1) * self._classes_per_regression, self._n_classes)

            X_test = base_scores[:, start_class:end_class, :].reshape((-1, n_base_models))
            preds = self._regressions[idx].predict(X_test)

            # Reshape predictions back to the original form and place them in the output array.
            preds_reshaped = preds.reshape(base_scores.shape[0], end_class - start_class)
            predictions[:, start_class:end_class] = preds_reshaped

        return predictions


# TODO: decide what to do about keeping this vs the one above.
class StackingMetaLearner_OLD:
    def __init__(self, n_classes: int):
        if n_classes < 1:
            raise ValueError('n_classes must be a positive integer.')
        self._n_classes = n_classes
        self._regressions = None

    def fit(self, base_scores: np.ndarray, labels: np.ndarray):
        """
        Base scores: (n_train_samples, n_classes, n_base_models)
        Labels: (n_train_samples, n_classes)
        """
        assert base_scores.ndim == 3 and labels.ndim == 2 \
               and base_scores.shape[0] == labels.shape[0] \
               and base_scores.shape[1] == self._n_classes \
               and labels.shape[1] == self._n_classes

        # Coefs may differ between classes.
        self._regressions = [LinearRegression(fit_intercept=False) for _ in range(self._n_classes)]

        for i in range(self._n_classes):
            self._regressions[i].fit(base_scores[:, i, :], labels[:, i])

    def predict(self, base_scores: np.ndarray) -> np.ndarray:
        assert self._n_classes is not None and isinstance(self._regressions, list)

        # Base scores: (n_samples, n_classes, n_base_models)
        assert base_scores.ndim == 3 and base_scores.shape[1] == self._n_classes

        # Predictions: (n_samples, n_classes)
        predictions = np.zeros((base_scores.shape[0], self._n_classes))
        for i in range(self._n_classes):
            predictions[:, i] = self._regressions[i].predict(base_scores[:, i, :])
        return predictions
