import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')
from helper_functions import load_procesed_data, setup_logger,add_border_and_title
from datasets import Lee2019
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
import os
import joblib
import time
import pandas as pd


def proc_epoch(epochs):
    # List of channels to pick
    channels_to_pick = [
        'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
    ]
    
    
    available_channels = epochs.ch_names
    
    # Pick only these channels from the epochs object
    channels_to_pick_filtered = [ch for ch in channels_to_pick if ch in available_channels]

    # Check if there are any channels left to pick
    if channels_to_pick_filtered:
        
        # Pick only the channels that exist
        epochs_proc = epochs.pick_channels(channels_to_pick_filtered)
        epochs_proc = epochs.filter(l_freq=8, h_freq=30, method='iir', iir_params=dict(order=5, ftype='butter'))
        return epochs_proc
    else:
        print("No matching channels found. Returning the original epochs object.")
        return epochs


def full_pipeline(epochs,subject=None,run=None,extra_mark= None, save=False,save_model=False, dataset_no=None, plot=False):
    dataset_no = str(dataset_no)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=3.5) #as in original article
    labels = epochs.events[:, -1] - 1 #original labels are 1,2
    
    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data(copy=False)
    epochs_data_train = epochs_train.get_data(copy=False)

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)
    
    # Printing the results
    std_score = np.std(scores)
    mean_score = np.mean(scores)
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)    
    print(f"Classification accuracy: {mean_score} / Chance level: {class_balance}")
    
    if save_model:
        if extra_mark == None:
            extra_mark = ''
        elif extra_mark != None:
            extra_mark = extra_mark + "_"            
        # Define the path to save the model
        path = os.path.join(os.getcwd(), 'data', 'models',dataset_no, f'{subject:02}')
        os.makedirs(path, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_save_path = os.path.join(
            path, 
            f's{subject:02}.{run:02}_{extra_mark}_{timestamp}_CSP_LDA_model.pkl'
        )
        
        # Save the fitted pipeline (CSP + LDA)
        joblib.dump(clf, model_save_path)
        print(f"Model saved to {model_save_path}")

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)
    if plot:
        csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    elif not plot:
        csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5,show=False)
        
    if save:
        if extra_mark == None:
            extra_mark = ''
        elif extra_mark != None:
            extra_mark = extra_mark + "_"
            
        path = os.path.join(os.getcwd(), 'data', 'plots', 'csp',dataset_no, f'{subject:02}')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f's{subject:02}.{run:02}_{extra_mark}CSP.png')
        plt.savefig(save_path)
        add_border_and_title(save_path, save_path,f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}",
                             border_size=0,title_color='black',title_bg_color='white')
        add_border_and_title(save_path, save_path, f's{subject:02}.{run:02}_{extra_mark}',
                             border_size=0,title_color='black',title_bg_color='white')
        

    sfreq = epochs.info["sfreq"]
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv.split(epochs_data_train):
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)
        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)
    ####
    # Convert scores to numpy array for easier manipulation
    scores_windows = np.array(scores_windows)  # Shape: (n_splits, n_windows)

    # Calculate mean and std over splits
    mean_scores_over_time = np.mean(scores_windows, axis=0)
    std_scores_over_time = np.std(scores_windows, axis=0)
    ####

    # Plot scores over time
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    plt.figure()
    # plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.plot(w_times, mean_scores_over_time, label="Score")
    plt.axvline(0, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    if plot:
        plt.show()
    
    if save:
        save_path = os.path.join(path, f's{subject:02}.{run:02}_{extra_mark}sliding_acc.png')
        plt.savefig(save_path)
        add_border_and_title(save_path, save_path, f's{subject:02}.{run:02}_{extra_mark}',
                             border_size=0,title_color='black',title_bg_color='white')
    
    results = {
        'subject': subject,
        'run': run,
        'extra_mark': extra_mark,
        'mean_score': mean_score,
        'std_score': std_score,
        'chance_level': class_balance,
        'w_times': w_times,
        'mean_scores_over_time': mean_scores_over_time,
        'std_scores_over_time': std_scores_over_time
    }
    return results


log = setup_logger("Lee_preprocess")

dataset = Lee2019()
dataset_no = 20
paradigm = "MI"

# Initialize lists to collect all results
all_results = []
all_over_time_results = []

for subject in dataset.subjects:
    for run in [1,2]:
        try:
            data = load_procesed_data(dataset_no, paradigm, subject, run, include=['epochs_raw', 'epochs_raw_autoreject'])
            
            epochs = data["epochs_raw"]
            epochs_autoreject = data["epochs_raw_autoreject"]
            
            epochs_p = proc_epoch(epochs)
            epochs_autoreject_p = proc_epoch(epochs_autoreject)
            
            result_epochs = full_pipeline(epochs_p, save = True,subject = subject, run=run,extra_mark='epochs',save_model=True, dataset_no='20')
            result_autoreject = full_pipeline(epochs_autoreject_p, save=True,subject = subject, run=run,extra_mark='autoreject',save_model=True, dataset_no='20')
            
            # Append overall results
            all_results.append({
                'subject': result_epochs['subject'],
                'run': result_epochs['run'],
                'extra_mark': result_epochs['extra_mark'],
                'mean_score': result_epochs['mean_score'],
                'std_score': result_epochs['std_score'],
                'chance_level': result_epochs['chance_level']
            })
            all_results.append({
                'subject': result_autoreject['subject'],
                'run': result_autoreject['run'],
                'extra_mark': result_autoreject['extra_mark'],
                'mean_score': result_autoreject['mean_score'],
                'std_score': result_autoreject['std_score'],
                'chance_level': result_autoreject['chance_level']
            })

            # Collect over-time results
            for idx, time_point in enumerate(result_epochs['w_times']):
                all_over_time_results.append({
                    'subject': subject,
                    'run': run,
                    'extra_mark': 'epochs',
                    'time': time_point,
                    'mean_score_over_time': result_epochs['mean_scores_over_time'][idx],
                    'std_score_over_time': result_epochs['std_scores_over_time'][idx]
                })
            for idx, time_point in enumerate(result_autoreject['w_times']):
                all_over_time_results.append({
                    'subject': subject,
                    'run': run,
                    'extra_mark': 'autoreject',
                    'time': time_point,
                    'mean_score_over_time': result_autoreject['mean_scores_over_time'][idx],
                    'std_score_over_time': result_autoreject['std_scores_over_time'][idx]
                })

            plt.close()
                        
        except Exception as e:
            # Log the error and continue with the next iteration
            log.error(f"Error processing subject {subject}, run {run}")
            log.error(f"Error details: {e}")
            continue

# Convert the list of over-time results into a DataFrame
over_time_results_df = pd.DataFrame(all_over_time_results)

# Convert overall results into a DataFrame
results_df = pd.DataFrame(all_results)

# Define paths to save the DataFrames
path_csv = os.path.join(os.getcwd(), 'data','models','20','stats')
os.makedirs(path_csv, exist_ok=True)
results_path = os.path.join(path_csv,'classification_results.csv')
over_time_results_path = os.path.join(path_csv,'classification_results_over_time.csv')

# Save the DataFrames
results_df.to_csv(results_path, index=False)
over_time_results_df.to_csv(over_time_results_path, index=False)
# Close plots