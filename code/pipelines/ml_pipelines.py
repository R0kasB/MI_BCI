import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')

from pipelines.ml_transformers import (
    ReshapeTransformer,
    PSDTransformer,
    ARTransformer,
    TimeDomainTransformer,
    HilbertTransformer,  
    WaveletTransformer,
    CARTransformer,
    STFTTransformer,
    MorletWaveletTransformer,
    MeanAmplitudeTransformer,
    FFTTransformer,
    SRCClassifier,
    ICATransformer,
    FilterBankTransformer,
    LogVarianceTransformer
)

# Import necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier, 
    StackingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from mne.decoding import CSP, Vectorizer
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.base import TransformerMixin
import numpy as np

__all__ = ['ml_pipelines', 'moabb_pipelines', 'all_ml_pipelines']

sfreq = 1000

ml_pipelines = {}
# 1. CSP + Logistic Regression
ml_pipelines['CSP_LR'] = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ('logreg', LogisticRegression(max_iter=sfreq))
])
# 2. PSD + SVM
ml_pipelines['PSD_SVM'] = Pipeline([
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
# 3. Time Domain Features + Random Forest
ml_pipelines['TimeDomain_RF'] = Pipeline([
    ('time_features', TimeDomainTransformer()),
    ('rf', RandomForestClassifier())
])
# 4. Hilbert Transform + k-NN #Neveikia
ml_pipelines['Hilbert_KNN'] = Pipeline([
    ('hilbert', HilbertTransformer()),
    ('reshape', ReshapeTransformer()),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
# 5. Wavelet Transform + PSD + Naive Bayes
ml_pipelines['Wavelet_PSD_NB'] = Pipeline([
    ('wavelet', WaveletTransformer()),
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('nb', GaussianNB())
])
# 6. CAR + CSP + Decision Tree
ml_pipelines['CAR_CSP_DT'] = Pipeline([
    ('car', CARTransformer()),
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ('dt', DecisionTreeClassifier())
])
# 7. ICA + Time Domain Features + SVM
ml_pipelines['ICA_TimeDomain_SVM'] = Pipeline([
    ('ica', ICATransformer(n_components=15, sfreq=sfreq)),
    ('time_features', TimeDomainTransformer()),
    ('svm', SVC())
])
# 8. CSP + LDA

#Yra moabb pipelines

# 9. PSD + Gradient Boosting
ml_pipelines['PSD_GB'] = Pipeline([
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier())
])
# 10. CAR + Riemannian Geometry Features + Logistic Regression #neveikia
ml_pipelines['CAR_Riemann_LR'] = Pipeline([
    ('car', CARTransformer()),
    ('cov', Covariances()),
    ('ts', TangentSpace()),
    ('logreg', LogisticRegression(max_iter=1000))
])
# 11. STFT + SVM
ml_pipelines['STFT_SVM'] = Pipeline([
    ('stft', STFTTransformer(n_fft=256)),
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
# 12. Morlet Wavelet Transform + k-NN 
#meta errorus, jei paprintint bandau ml_pipelines dėl šito pipelino
ml_pipelines['Morlet_KNN'] = Pipeline([
    ('morlet', MorletWaveletTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
# 13. AR Coefficients + Logistic Regression
ml_pipelines['AR_LR'] = Pipeline([
    ('ar', ARTransformer(order=5)),
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])
# 14. Mean Amplitude Features + Naive Bayes
ml_pipelines['MeanAmp_NB'] = Pipeline([
    ('mean_amp', MeanAmplitudeTransformer()),
    ('nb', GaussianNB())
])
# 15. ICA + PSD + SVM
ml_pipelines['ICA_PSD_SVM'] = Pipeline([
    ('ica', ICATransformer(n_components=15, sfreq=sfreq)),
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
# 16. FFT + Random Forest
ml_pipelines['FFT_RF'] = Pipeline([
    ('fft', FFTTransformer()),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])
# 17. Time-Frequency Features (STFT) + LDA
ml_pipelines['TimeFreq_LDA'] = Pipeline([
    ('stft', STFTTransformer(n_fft=256)),
    ('scaler', StandardScaler()),
    ('lda', LDA())
])
# 18. PSD Features + XGBoost Classifier
ml_pipelines['PSD_XGB'] = Pipeline([
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])
# 19. Ensemble Methods (Stacking Classifier)
ml_pipelines['Stacking'] = Pipeline([
    ('psd', PSDTransformer(sfreq=sfreq)),
    ('scaler', StandardScaler()),
    ('stacking', StackingClassifier(
        estimators=[
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier()),
            ('knn', KNeighborsClassifier())
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    ))
])
# 20. Sparse Representation Classification (SRC)
ml_pipelines['SRC'] = Pipeline([
    ('reshape', ReshapeTransformer()),
    ('scaler', StandardScaler()),
    ('src', SRCClassifier(alpha=0.1))
])
# 21. Multilayer Perceptron (MLP) Neural Network (As ML Pipeline)
ml_pipelines['MLP_NN'] = Pipeline([
    ('reshape', ReshapeTransformer()),
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500))
])

# Pipeline dictionary
moabb_pipelines = {}

# 1. ACM + Tangent Space + SVM
moabb_pipelines['ACM_TS_SVM'] = Pipeline([
    ('cov', Covariances()),  # Adaptive CSP could be represented using covariance estimation
    ('ts', TangentSpace()),  # Tangent Space transformation
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC())  # SVM classifier
])

# 2. CSP + SVM
moabb_pipelines['CSP_SVM'] = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),  # CSP
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC())  # SVM classifier
])

# 3. DLCSPauto + Shrinkage LDA
moabb_pipelines['DLCSPauto_shLDA'] = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),  # Simplified CSP auto
    ('scaler', StandardScaler()),  # Feature scaling
    ('lda', LDA(solver='lsqr', shrinkage='auto'))  # LDA with shrinkage
])

# 4. FilterBank + SVM
moabb_pipelines['FilterBank_SVM'] = Pipeline([
    ('filter_bank', FilterBankTransformer(estimator=CSP(n_components=4), 
                                          filters=[(8, 12), (12, 30)], sfreq=sfreq, flatten=True)),  # Apply FilterBankTransformer with CSP
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC())  # SVM classifier
])

# 5. FgMDM (Geodesic Filtering with MDM)
moabb_pipelines['FgMDM'] = Pipeline([
    ('cov', Covariances()),  # Covariance estimation
    ('mdm', MDM())  # MDM classifier from pyriemann
])

# 6. LogVariance + LDA
moabb_pipelines['LogVariance_LDA'] = Pipeline([
    ('logvar', LogVarianceTransformer()),  # Log-variance feature extraction
    ('scaler', StandardScaler()),  # Feature scaling
    ('lda', LDA())  # LDA classifier
])

# 7. LogVariance + SVM
moabb_pipelines['LogVariance_SVM'] = Pipeline([
    ('logvar', LogVarianceTransformer()),  # Log-variance feature extraction
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC())  # SVM classifier
])

# 8. MDM (Minimum Distance to Mean)
moabb_pipelines['MDM'] = Pipeline([
    ('cov', Covariances()),  # Covariance estimation
    ('mdm', MDM())  # MDM classifier from pyriemann
])

# 9. TRCSP + LDA


# 10. Tangent Space + Ensemble Learning (EL)
moabb_pipelines['TS_EL'] = Pipeline([
    ('cov', Covariances()),  # Covariance estimation
    ('ts', TangentSpace()),  # Tangent Space transformation
    ('scaler', StandardScaler()),  # Feature scaling
    ('ensemble', RandomForestClassifier())  # RandomForest as an ensemble method
])

# 11. Tangent Space + Logistic Regression (LR)
moabb_pipelines['TS_LR'] = Pipeline([
    ('cov', Covariances()),  # Covariance estimation
    ('ts', TangentSpace()),  # Tangent Space transformation
    ('scaler', StandardScaler()),  # Feature scaling
    ('logreg', LogisticRegression(max_iter=1000))  # Logistic Regression
])

# 12. Tangent Space + SVM
moabb_pipelines['TS_SVM'] = Pipeline([
    ('cov', Covariances()),  # Covariance estimation
    ('ts', TangentSpace()),  # Tangent Space transformation
    ('scaler', StandardScaler()),  # Feature scaling
    ('svm', SVC())  # SVM classifier
])

# 13. CSP + LDA
moabb_pipelines['CSP_LDA'] = Pipeline([
    ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ('lda', LDA())
])


# Merge all pipelines at the bottom
all_ml_pipelines = {**moabb_pipelines, **ml_pipelines}