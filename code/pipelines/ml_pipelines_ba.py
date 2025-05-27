import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')

# Import necessary libraries
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from pipelines.ml_transformers import (
    PSDTransformer,
    TimeDomainTransformer,
    STFTTransformer,
    LogVarianceTransformer,
    FBCSPTransformer,
    CSSPTransformer,
    BSSFOTransformer
)


__all__ = ['features', 'classifiers']

sfreq = 1000

features = {}
features['CSP']  = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False))]) # visiem buvo reg=False, cleaned neisejo tai su reg 0,1
# https://stackoverflow.com/questions/76431070/mne-valueerror-data-copying-was-not-requested-by-copy-none-but-it-was-require
# jei i auto nepakeiciu, nefitina

features['CSSP'] = Pipeline([
    ('cssp', CSSPTransformer(n_components=4, l_freq=8, h_freq=30, sfreq=sfreq))])

features['BSSFO'] = Pipeline([
    ('bssfo', BSSFOTransformer(n_components=20, random_state=42))])

features['FBCSP'] = Pipeline([
('fbcsp', FBCSPTransformer(filters=[(8, 12), (12, 16), (16, 20)],
                           sfreq=sfreq, n_components=4, flatten=True))])

features['PSD'] = Pipeline([
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler())])

features['TDF'] = Pipeline([
    ('time_features', TimeDomainTransformer())])

features['STFT'] = Pipeline([
    ('stft', STFTTransformer(n_fft=256)),
    ('scaler', StandardScaler())])

features['LogVar'] = Pipeline([
    ('logvar', LogVarianceTransformer()),  # Log-variance feature extraction
    ('scaler', StandardScaler())])

features['TS'] = Pipeline([
    ('cov', Covariances(estimator='oas')),  # Covariance estimation
    ('ts', TangentSpace())  # Tangent Space transformation
    # ('scaler', StandardScaler())
    ])

classifiers= {}

classifiers['SVM'] = Pipeline([
    ('SVM', SVC())])

classifiers['LR'] = Pipeline([
    ('LR', LogisticRegression(max_iter=1000))])

classifiers['LDA'] = Pipeline([
    ('LDA', LDA())])

classifiers['RF'] = Pipeline([
    ('RF', RandomForestClassifier())])  