import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'code'))

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
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False))]) 

features['CSSP'] = Pipeline([
    ('cssp', CSSPTransformer(n_components=4, delays=(0,2,4,6)))])

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
    ('logvar', LogVarianceTransformer()), 
    ('scaler', StandardScaler())])

features['TS'] = Pipeline([
    ('cov', Covariances(estimator='oas')),  
    ('ts', TangentSpace())  
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