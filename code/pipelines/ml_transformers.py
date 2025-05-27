import numpy as np
import mne
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import Lasso
from mne import filter
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import FastICA
from sklearn.base import BaseEstimator, TransformerMixin

class ReshapeTransformer(BaseEstimator, TransformerMixin):
    """Reshapes 3D EEG data to 2D for classifier input."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)
class PSDTransformer(BaseEstimator, TransformerMixin):
    """Extracts Power Spectral Density (PSD) features."""

    def __init__(self, sfreq=256, fmin=8, fmax=30):
        # Initialize with default values
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
    def fit(self, X, y=None):
        # Fit method can be used to define sfreq, fmin, fmax dynamically later
        return self
    def transform(self, X):
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            X, sfreq=self.sfreq, fmin=self.fmin, fmax=self.fmax, verbose=False
        )
        return psd.reshape(psd.shape[0], -1)

class ARTransformer(BaseEstimator, TransformerMixin):
    """Computes Autoregressive (AR) coefficients."""
    def __init__(self, order=5):
        self.order = order
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        from statsmodels.tsa.ar_model import AutoReg
        n_samples, n_channels, n_times = X.shape
        ar_features = np.zeros((n_samples, n_channels * self.order))
        for i in range(n_samples):
            for j in range(n_channels):
                model = AutoReg(X[i, j, :], lags=self.order, old_names=False)
                model_fit = model.fit()
                ar_coeffs = model_fit.params[1:]  # Exclude intercept
                ar_features[i, j * self.order:(j + 1) * self.order] = ar_coeffs
        return ar_features
    
class TimeDomainTransformer(BaseEstimator, TransformerMixin):
    """
    Computes time-domain features for each channel:
      - Mean, Mean Amplitude, Variance, Median, RMS, IQR, Zero-crossing Rate.
    
    Expects input X with shape (n_samples, n_channels, n_times).
    """
    def __init__(self):
        # List of feature names corresponding to the computed features.
        self.metric_names = ["mean", "mean_amplitude", "variance", "median", "rms", "iqr", "zcr"]

    def fit(self, X, y=None):
        return self

    def _zero_crossing_rate(self, signal):
        # Compute the zero-crossing rate for a given signal.
        return np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * len(signal))

    def transform(self, X):
        features_list = []
        for sample in X:
            sample_features = []
            for channel in sample:
                mu = np.mean(channel)
                mu_amp = np.mean(np.abs(channel))
                var = np.var(channel)
                med = np.median(channel)
                rms = np.sqrt(np.mean(channel ** 2))
                iqr = np.percentile(channel, 75) - np.percentile(channel, 25)
                zcr = self._zero_crossing_rate(channel)
                feats = [mu, mu_amp, var, med, rms, iqr, zcr]
                feats = np.array(feats)
                if np.isnan(feats).any():
                    for name, value in zip(self.metric_names, feats):
                        if np.isnan(value):
                            print(f"NaN detected for metric '{name}' in channel: {channel}")
                sample_features.append(feats)
            features_list.append(np.concatenate(sample_features))
        return np.array(features_list)

        
class MeanAmplitudeTransformer(BaseEstimator, TransformerMixin):
    """Computes the mean amplitude of the signal."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        mean_amplitude = np.mean(np.abs(X), axis=2)
        return mean_amplitude
    
class HilbertTransformer(BaseEstimator, TransformerMixin):
    """Applies the Hilbert transform to EEG data."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        analytic_signal = mne.filter.hilbert(X, picks=None, envelope=False, verbose=False)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope
    
class WaveletTransformer(BaseEstimator, TransformerMixin):
    """Applies Wavelet Transform."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        import pywt
        n_samples, n_channels, n_times = X.shape
        wavelet_features = []
        for i in range(n_samples):
            sample_features = []
            for j in range(n_channels):
                coeffs = pywt.wavedec(X[i, j, :], 'db4', level=3)
                coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
                sample_features.append(coeffs_flat)
            wavelet_features.append(np.concatenate(sample_features))
        return np.array(wavelet_features)

class CARTransformer(BaseEstimator, TransformerMixin):
    """Applies Common Average Referencing (CAR)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        car = X - np.mean(X, axis=1, keepdims=True)
        return car
    
class ICATransformer(BaseEstimator, TransformerMixin):
    """Applies Independent Component Analysis (ICA)."""
    def __init__(self, n_components=15, random_state=42, sfreq=256):
        self.n_components = n_components
        self.random_state = random_state
        self.ica = None
        self.sfreq = sfreq  # Default to 256, but can be changed
        self.info = None
    def fit(self, X, y=None):
        n_samples, n_channels, n_times = X.shape
        X_concat = np.concatenate(X, axis=-1)  # Concatenate along the time axis, shape becomes (n_channels, n_total_times)
        self.info = mne.create_info(ch_names=['eeg'] * n_channels, sfreq=self.sfreq, ch_types='eeg')
        self.ica = mne.preprocessing.ICA(n_components=self.n_components,
                                         random_state=self.random_state,
                                         max_iter='auto', verbose=False)
        raw = mne.io.RawArray(X_concat, self.info, verbose=False)
        self.ica.fit(raw)
        return self
    
    def transform(self, X):
        n_samples, n_channels, n_times = X.shape
        X_transformed = []
        for i in range(n_samples):
            data = X[i]
            raw = mne.io.RawArray(data, self.info, verbose=False)
            raw_ica = self.ica.apply(raw.copy(), exclude=[], verbose=False)
            X_transformed.append(raw_ica.get_data())
        return np.array(X_transformed)
        
class STFTTransformer(BaseEstimator, TransformerMixin):
    """Computes Short-Time Fourier Transform (STFT)."""
    def __init__(self, n_fft=256, nperseg=256, noverlap=None):
        """
        Parameters:
        ------------
        n_fft : int
            Number of points in the FFT.
        nperseg : int
            Length of each segment for STFT.
        noverlap : int, optional
            Number of points to overlap between segments.
        """
        self.n_fft = n_fft
        self.nperseg = nperseg
        self.noverlap = noverlap
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        from scipy.signal import stft
        n_samples, n_channels, n_times = X.shape
        stft_features = []
        for i in range(n_samples):
            sample_features = []
            for j in range(n_channels):
                _, _, Zxx = stft(X[i, j, :], nperseg=self.nperseg, nfft=self.n_fft, noverlap=self.noverlap)
                sample_features.append(np.abs(Zxx).flatten())
            stft_features.append(np.concatenate(sample_features))
        return np.array(stft_features)

class MorletWaveletTransformer(BaseEstimator, TransformerMixin):
    """Applies time-frequency analysis using Morlet wavelets."""
    def __init__(self, sfreq=256,freqs=np.linspace(8, 30, num=22), n_cycles=7):
        self.sfreq = sfreq
        self.freqs = freqs
        self.n_cycles = n_cycles
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        n_samples, n_channels, n_times = X.shape
        power_features = []
        for i in range(n_samples):
            sample_features = []
            for j in range(n_channels):
                power = mne.time_frequency.tfr_array_morlet(
                    X[i:i+1, j:j+1, :], sfreq=self.sfreq, freqs=self.freqs,
                    n_cycles=self.n_cycles, output='power', verbose=False)
                sample_features.append(power.flatten())
            power_features.append(np.concatenate(sample_features))
        return np.array(power_features)
        
class FFTTransformer(BaseEstimator, TransformerMixin):
    """Computes Fast Fourier Transform (FFT)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        fft_coeffs = np.fft.rfft(X, axis=2)
        fft_features = np.abs(fft_coeffs)
        return fft_features.reshape(X.shape[0], -1)

class SRCClassifier(BaseEstimator, ClassifierMixin):
    """Simulates a Sparse Representation Classifier using Lasso."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.classes_ = None
        self.dictionary_ = None
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.dictionary_ = {}
        for cls in self.classes_:
            self.dictionary_[cls] = X[y == cls].T
        return self
    def predict(self, X):
        preds = []
        for x in X:
            residuals = []
            for cls in self.classes_:
                lasso = Lasso(alpha=self.alpha, max_iter=1000)
                lasso.fit(self.dictionary_[cls], x)
                reconstruction = lasso.predict(self.dictionary_[cls])
                residual = np.linalg.norm(x - reconstruction)
                residuals.append(residual)
            preds.append(self.classes_[np.argmin(residuals)])
        return np.array(preds)

class LogVarianceTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        epsilon = 1e-10  # A very small constant (kitu atveju kai variance 0, returnina infinity)
        return np.log(np.var(X, axis=-1) + epsilon)


class FilterBankTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a filter bank to the input data and then apply the same estimator (e.g., CSP)
    to each filtered signal. Optionally concatenate or stack the resulting features.

    Parameters:
    ----------
    estimator : sklearn-compatible transformer or estimator
        The estimator (e.g., CSP) to apply on each filtered signal.
    
    filters : list of tuples
        List of frequency bands (tuples) to apply as bandpass filters.
        Example: [(8, 12), (12, 30)] would apply two bandpass filters, one from 8-12 Hz and one from 12-30 Hz.
    
    sfreq : float
        Sampling frequency of the signal.
    
    flatten : bool, optional (default=True)
        If True, flatten the features from each band and concatenate. If False, keep them as separate features.
    """
    
    def __init__(self, estimator, filters, sfreq, flatten=True):
        self.estimator = estimator
        self.filters = filters
        self.sfreq = sfreq
        self.flatten = flatten
        self.estimators_ = []

    def fit(self, X, y=None):
        # Fit the estimator for each frequency band
        self.estimators_ = []
        for fmin, fmax in self.filters:
            filtered_data = self._bandpass_filter(X, fmin, fmax)
            est = clone(self.estimator)
            est.fit(filtered_data, y)
            self.estimators_.append(est)
        return self

    def transform(self, X):
        # Apply each estimator to the corresponding filtered data
        transformed_data = []
        for (fmin, fmax), est in zip(self.filters, self.estimators_):
            filtered_data = self._bandpass_filter(X, fmin, fmax)
            transformed = est.transform(filtered_data)
            transformed_data.append(transformed)

        if self.flatten:
            # Flatten and concatenate features from each frequency band
            return np.concatenate(transformed_data, axis=1)
        else:
            # Stack the features along a new axis
            return np.stack(transformed_data, axis=-1)

    def _bandpass_filter(self, X, fmin, fmax):
        # Apply a bandpass filter using MNE's filter function
        return filter.filter_data(X, sfreq=self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class BandpassFilter(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a bandpass filter to each trial using MNE.
    Assumes input X is a NumPy array of shape (n_trials, n_channels, n_times).
    """
    def __init__(self, l_freq, h_freq, sfreq):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        filtered = [mne.filter.filter_data(trial, self.sfreq, self.l_freq,
                                             self.h_freq, verbose=False) for trial in X]
        return np.array(filtered)
    
class CSSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, delays=(0,2,4,6), n_components=4):
        self.delays = delays
        self.n_components = n_components

    def _embed(self, X):
        n_trials, n_ch, n_times = X.shape
        L = len(self.delays)
        X_aug = np.zeros((n_trials, n_ch*L, n_times))
        for i, trial in enumerate(X):
            mats = []
            for d in self.delays:
                pad = np.pad(trial, ((0,0),(d,0)), mode='constant')[:, :n_times]
                mats.append(pad)
            X_aug[i] = np.vstack(mats)
        return X_aug

    def fit(self, X, y):
        X = np.array(X)  # shape (n_trials, n_ch, n_times)
        X_aug = self._embed(X)
        self.csp_ = CSP(n_components=self.n_components, log=True)
        self.csp_.fit(X_aug, y)
        return self

    def transform(self, X):
        X = np.array(X)
        X_aug = self._embed(X)
        return self.csp_.transform(X_aug)


class BSSFOTransformer(BaseEstimator, TransformerMixin):
    """
    BSSFO-style transformer using Blind Source Separation.
    Here we use scikit-learn's FastICA to extract independent components,
    then compute the log-variance of each source as features.
    """
    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        # X shape: (n_trials, n_channels, n_times)
        n_trials, n_channels, n_times = X.shape
        # Concatenate trials along the time axis
        X_concat = X.transpose(0, 2, 1).reshape(-1, n_channels)
        self.ica_ = FastICA(n_components=self.n_components, random_state=self.random_state)
        self.ica_.fit(X_concat)
        return self

    def transform(self, X):
        features = []
        for trial in X:
            # trial shape: (n_channels, n_times)
            trial = trial.T  # shape becomes (n_times, n_channels)
            sources = self.ica_.transform(trial)  # shape (n_times, n_components)
            sources = sources.T  # shape (n_components, n_times)
            # Compute log-variance for each component (add a small epsilon to avoid log(0))
            trial_features = np.log(np.var(sources, axis=1) + 1e-10)
            features.append(trial_features)
        return np.array(features)

class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for Filter Bank Common Spatial Patterns (FBCSP).
    For each frequency band, it applies a bandpass filter, fits a CSP estimator, 
    and then transforms the data. The features across bands are concatenated.

    Parameters
    ----------
    filters : list of tuple
        List of frequency bands as (l_freq, h_freq) tuples.
    sfreq : float
        Sampling frequency of the EEG data.
    n_components : int, default=4
        Number of CSP components per band.
    flatten : bool, default=True
        If True, features are flattened and concatenated.
    """
    
    def __init__(self, filters, sfreq, n_components=4, flatten=True):
        self.filters = filters
        self.sfreq = sfreq
        self.n_components = n_components
        self.flatten = flatten

    def fit(self, X, y):
        """
        Fit a CSP estimator for each frequency band.

        Parameters
        ----------
        X : array-like, shape (n_trials, n_channels, n_times)
            Input EEG data.
        y : array-like, shape (n_trials,)
            Class labels.
        
        Returns
        -------
        self
        """
        self.csp_estimators_ = []
        for (l_freq, h_freq) in self.filters:
            # Apply bandpass filtering to each trial for the current band
            X_filt = np.array([mne.filter.filter_data(trial, self.sfreq, l_freq, h_freq, verbose=False)
                               for trial in X])
            csp = CSP(n_components=self.n_components, log=True) #kai su cleaned runina reikia reg=0.1
            csp.fit(X_filt, y)
            self.csp_estimators_.append(csp)
        return self

    def transform(self, X):
        """
        Transform the data by applying band-specific CSP and concatenating features.

        Parameters
        ----------
        X : array-like, shape (n_trials, n_channels, n_times)
            Input EEG data.
        
        Returns
        -------
        features : array-like
            Transformed feature array. If flatten is True, the output has shape 
            (n_trials, n_components * n_bands); otherwise, shape (n_trials, n_bands, n_components).
        """
        features = []
        for (l_freq, h_freq), csp in zip(self.filters, self.csp_estimators_):
            X_filt = np.array([mne.filter.filter_data(trial, self.sfreq, l_freq, h_freq, verbose=False)
                               for trial in X])
            features.append(csp.transform(X_filt))
        if self.flatten:
            return np.concatenate(features, axis=1)
        else:
            return np.stack(features, axis=1)
