import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your EEG data (replace this with actual file path)
raw = mne.io.read_raw_fif('your_data_file.fif', preload=True)

# 1. Select motor cortex channels
motor_cortex_channels = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
                         'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                         'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']

raw.pick_channels(motor_cortex_channels)

# 2. Band-pass filter the data (8-30 Hz, 5th order Butterworth filter)
raw.filter(8., 30., fir_design='butterworth', order=5)

# 3. Epoch the data (1000 ms to 3500 ms after stimulus onset)
# Assuming you have events and event IDs, replace 'event_id' and 'events'
events = mne.find_events(raw)  # You should have events in your dataset
event_id = {'left_hand': 1, 'right_hand': 2}  # Replace with your actual event IDs

# Epochs from 1s to 3.5s (1000 ms to 3500 ms)
epochs = mne.Epochs(raw, events, event_id, tmin=1.0, tmax=3.5, preload=True)

# Get the epoch data and labels
X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # Labels (1 for left-hand, 2 for right-hand)

# 4. CSP - Common Spatial Patterns
# Initialize CSP with 4 components (2 top, 2 bottom)
csp = CSP(n_components=4, reg=None, log=None, cov_est='epoch')

# 5. Log-variance features
# Build a pipeline: first CSP, then LDA
lda = LinearDiscriminantAnalysis()

# 6. Train LDA classifier
# Create a pipeline with CSP and LDA
csp_lda_pipeline = make_pipeline(csp, lda)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the pipeline on the training data
csp_lda_pipeline.fit(X_train, y_train)

# 7. Evaluate the classifier
# Predict on the test data
y_pred = csp_lda_pipeline.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the CSP patterns
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
plt.show()

# Optional: Extract top and bottom two rows from CSP matrix for further use
csp_patterns = csp.filters_  # CSP filters matrix
top_two_csp = csp_patterns[:2, :]
bottom_two_csp = csp_patterns[-2:, :]
