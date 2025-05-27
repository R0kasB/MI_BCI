import sys
import os
import pickle
from datetime import datetime
import time
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler


sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')
from pipelines.ml_pipelines import all_ml_pipelines
from evaluation import train_and_evaluate
from helper_functions import setup_logger, load_procesed_data, process_mi_epochs
from datasets import Lee2019

# Define the folder for the checkpoint
CHECKPOINT_FOLDER = os.path.join("data", "checkpoints")
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
CHECKPOINT_FILE = os.path.join(CHECKPOINT_FOLDER, "checkpoint_fine_tune_01.pkl")

# Define the runtime log file
RUNTIME_LOG_DIR = os.path.join("data", "logs")
os.makedirs(RUNTIME_LOG_DIR, exist_ok=True)  # Ensure logs directory exists
RUNTIME_LOG_FILE = os.path.join(RUNTIME_LOG_DIR, "pipeline_runtime_log_TS_EL_experiment.csv")

# Ensure the runtime log file exists and create headers if not
if not os.path.exists(RUNTIME_LOG_FILE):
    with open(RUNTIME_LOG_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pipeline", "Subject", "Run", "Runtime (seconds)"])

'''
Pipeline,Subject,Run,Runtime (seconds)
TS_EL,3,1,61761.05

param_grids = {
    'TS_EL' : {
        'ensemble__n_estimators': [100, 200, 350, 500, 1000],
        'ensemble__max_depth': [None, 1,5, 10, 20, 30],
        'ensemble__min_samples_split': [2, 5, 10],
        'ensemble__min_samples_leaf': [1, 2, 4],
        'ensemble__class_weight': [None, 'balanced'],
        'ensemble__max_features': ['auto', 'sqrt', 'log2'],
        'ensemble__bootstrap': [True, False],
        'cov__estimator': ['oas', 'lwf', 'scm'],
        'ts__metric': ['riemann', 'euclidean'],
        'scaler': [StandardScaler(), MinMaxScaler(), None],
        'ensemble__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
        },
    } 

'''

param_grids = {
    'TRCSP_LDA': {
        'csp__n_components': [2, 4, 6, 8, 10, 12, 14, 16],  # Expanded range
        'lda__shrinkage': ['auto',None, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Slightly expanded
        'lda__solver': ['lsqr', 'eigen', 'svd'],  # Kept original
        'lda__tol': [1e-4, 1e-2],  # Reduced to 2 tolerance values
    },
    'TS_EL': {
        'ensemble__n_estimators': [100, 200, 300, 400],
        'ensemble__max_depth': [None, 5, 10, 30],
        'ensemble__min_samples_split': [2, 5, 10],
        'ensemble__min_samples_leaf': [1, 2, 4],
    },    
    'TS_LR': {
        'logreg__C': [0.001, 0.01, 0.1, 1, 10, 500],
        'logreg__solver': ['lbfgs', 'liblinear', 'saga'],  # Kept original
        'logreg__max_iter': [100, 200, 300, 500],
        'logreg__penalty': ['l2', 'l1', 'elasticnet', 'none'],
    },
    'TS_SVM': {
        'svm__C': [0.01, 0.1, 1, 5, 10, 100, 500, 1000],  # Expanded range
        'svm__kernel': ['linear', 'rbf', 'poly'],  # Added polynomial kernel
        'svm__gamma': ['scale', 'auto', 0.1, 0.3, 0.5, 0.8, 1],  # Expanded granularity
    },
    'LogVariance_LDA': {
        'lda__shrinkage': ['auto', None, 0.1, 0.5, 0.9],  # Reduced to fewer key values
        'lda__solver': ['lsqr', 'eigen','svd'],  # Reduced to 2 solvers
        'lda__tol': [1e-4, 1e-2],  # Reduced to 2 tolerance values
        'lda__n_components': [None, 2],  # Reduced to 2 values
        'lda__priors': [None, 'uniform'],  # Fixed to None to eliminate a dimension
        'lda__store_covariance': [True, False],  # Fixed to True for simplicity  
    },
    'CAR_CSP_DT': {
        'csp__n_components': [2, 4, 6, 8, 10],
        'csp__reg': [None, 'ledoit_wolf', 'oas', 'shrunk'],
        'dt__max_depth': [10, 15, None],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2],
    },
    'CSP_LDA': {
        'csp__n_components': [2, 4, 6, 8, 10, 12, 14],
        'csp__reg': [None, 'ledoit_wolf', 'oas', 'shrunk'],
        'lda__shrinkage': ['auto', None, 0.1, 0.3, 0.5, 0.8],
        'lda__solver': ['lsqr', 'eigen'],
    },
    'CSP_LogReg': {
        'csp__n_components': [2, 4, 6, 8, 10],
        'csp__reg': [None, 'ledoit_wolf', 'oas', 'shrunk'],
        'logreg__C': [0.01, 0.1, 1, 10, 100],
        'logreg__solver': ['lbfgs', 'liblinear', 'saga'],
        'logreg__max_iter': [100, 200],
    },
    'CSP_SVM': {
        'csp__n_components': [2, 4, 6, 8, 10],
        'csp__reg': [None, 'ledoit_wolf', 'oas'],
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto', 0.1, 1],
    },
    'DLCSPauto_shLDA': {
        'lda__shrinkage': ['auto', None, 0.1, 0.3, 0.5, 0.7, 0.9],  # Focused on key shrinkage values
        'lda__solver': ['lsqr', 'eigen', 'svd'],  # Commonly used solvers
        'lda__tol': [1e-4, 1e-2, 1e-1],  # Reduced granularity
        'lda__n_components': [None, 1, 2, 5],  # Focused on typical values
        'lda__priors': [None, 'uniform'],  # Retained class priors
        'lda__store_covariance': [True],  # Fixed to True for simplicity 
    },
    'ACM_TS_SVM': {
        'svm__C': [0.01, 0.1, 1, 10, 500, 1000],  # Expanded range
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kept original
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Expanded granularity
        'svm__degree': [2, 3, 5],  # Kept original
    },
}

def save_checkpoint(subject, run, pipeline_key):
    """Save the current progress to a checkpoint file."""
    checkpoint = {"subject": subject, "run": run, "pipeline_key": pipeline_key}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {checkpoint}")

def load_checkpoint():
    """Load progress from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    # Set up logger
    log = setup_logger("Lee_ml_pipeline_run", log_file=os.path.join("logs", "Lee_full_dataset_ml_run.log"))
    dataset = Lee2019()
    dataset_no = 20
    paradigm = "MI"

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_subject = checkpoint['subject'] if checkpoint else 1
    start_run = checkpoint['run'] if checkpoint else 1
    start_pipeline_key = checkpoint['pipeline_key'] if checkpoint else None

    try:
        for subject in range(start_subject, 55):
            if subject != start_subject:
                start_run = 1
            for run in range(start_run, 3):
                pipeline_keys = list(all_ml_pipelines.keys())
                if start_pipeline_key and subject == start_subject and run == start_run:
                    pipeline_keys = pipeline_keys[pipeline_keys.index(start_pipeline_key)+1:]
                    start_pipeline_key = None

                for pipeline_key in pipeline_keys:
                    if pipeline_key not in param_grids:  # Skip pipelines without a param_grid
                        continue
                    
                    pipeline = all_ml_pipelines[pipeline_key]
                    param_grid = param_grids.get(pipeline_key, None)
                    try:
                        data = load_procesed_data(dataset_no, paradigm, subject, run, include=['epochs_raw'])
                        epochs = data["epochs_raw"]
                        epochs_p = process_mi_epochs(epochs)
                        
                        start_time = time.time()  # Start timer

                        results = train_and_evaluate(
                            epochs_p, 
                            pipeline, 
                            n_splits=5, 
                            log=log, 
                            save=True, 
                            single_pipeline_key=pipeline_key,
                            param_grid=param_grid,  # Pass the param grid
                            subject=subject,
                            run=run,
                            n_jobs=-1
                        )

                        end_time = time.time()  # End timer
                        runtime = end_time - start_time  # Calculate runtime
                        log.info(f"Pipeline {pipeline_key}, Subject {subject}, Run {run} completed in {runtime:.2f} seconds.")
                        
                        # Save runtime to CSV log file
                        with open(RUNTIME_LOG_FILE, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([pipeline_key, subject, run, f"{runtime:.2f}"])

                        directory = os.path.join(os.getcwd(), 'data', 'results', 'fine_tune_scores', 'lee2019', '2025-01-05')
                        os.makedirs(directory, exist_ok=True)
                        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                        file_name = os.path.join(directory, f'{pipeline_key}_result_{subject:02}-{run:02}_{current_time}.pkl')
                        with open(file_name, 'wb') as f:
                            pickle.dump(results, f)
                            log.info(f"{file_name} - saved")
                                                    

                    except Exception as e:
                        log.error(f"Error processing subject {subject}, run {run}, pipeline {pipeline_key}")
                        log.error(f"Error details: {e}")
                    
                    # Save checkpoint after processing each pipeline
                    save_checkpoint(subject, run, pipeline_key)
    except KeyboardInterrupt:
        print("Interrupted. Progress saved.")
    except Exception as e:
        print(f"An error occurred: {e}. Progress saved.")

if __name__ == "__main__":
    main()
