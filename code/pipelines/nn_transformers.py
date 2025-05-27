from sklearn.base import BaseEstimator, TransformerMixin


class EEGNetTransformer(BaseEstimator, TransformerMixin):
    """Transformer that reshapes data for EEGNet."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Reshape X to (n_samples, n_channels, n_times, 1)
        n_samples, n_channels, n_times = X.shape
        X_reshaped = X.reshape(n_samples, n_channels, n_times, 1)
        return X_reshaped

