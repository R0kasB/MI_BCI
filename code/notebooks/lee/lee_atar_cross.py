import sys
import os
import pickle
from datetime import datetime
from joblib import Parallel, delayed

# add your code directory to the path
sys.path.append(r'c:\Users\rokas\Documents\GitHub\BCI\mi-bci\code')

from evaluation import cs_evaluate
from helper_functions import setup_logger, load_procesed_data
from pipelines.ml_pipelines_ba import features, classifiers

# constants
dataset_no  = 20
paradigm    = "MI"
extra_mark  = "(-5_10)(8_30Hz)"
log_dir     = "logs"

def run_subject(subject: int, addon: str):
    """Load both runs for one subject/addon, run cs_evaluate, and save the results."""
    # ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"subj{subject:02}_{addon}.log")
    log = setup_logger(f"run_subj{subject:02}_{addon}", log_file=log_file)

    try:
        # --- load run 1 & 2 epochs ---
        d1 = load_procesed_data(
            dataset_no, paradigm, subject, run=1,
            include=[f"epochs_atar_{addon}"], extra_mark=extra_mark
        )
        d2 = load_procesed_data(
            dataset_no, paradigm, subject, run=2,
            include=[f"epochs_atar_{addon}"], extra_mark=extra_mark
        )
        epochs = {
            "1": d1[f"epochs_atar_{addon}"],
            "2": d2[f"epochs_atar_{addon}"]
        }

        # --- evaluate cross- and within-session ---
        results = cs_evaluate(
            epochs=epochs,
            n_splits=5,
            save_model=True,
            log=log,
            subject=subject,
            construct_pipelines=True,
            feature_pipelines=features,
            classifier_pipelines=classifiers,
            n_jobs=1,  # avoid nested parallelism
            save_dir=os.path.join("models", f"cs_eval_atar_{addon}"),
            train_run=1,
            test_run=2,
            return_within=True
        )

        # --- save pickle ---
        out_dir = os.path.join(
            os.getcwd(), "data", "results", "pipeline_scores",
            "lee2019", f"2025-04-22_atar_{addon}"
        )
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = os.path.join(
            out_dir,
            f"cs_eval_result_atar_{addon}_{subject:02}_{ts}.pkl"
        )
        with open(fname, "wb") as f:
            pickle.dump(results, f)
        log.info(f"[Subject {subject:02}] Saved results to {fname}")

    except Exception as e:
        log.error(f"[Subject {subject:02}] Error: {e}")

# list of addon variants
addons = [
    # 'beta0.01_OptModeelim',   'beta0.01_OptModelinAtten', 'beta0.01_OptModesoft',
    # 'beta0.5_OptModeelim',    'beta0.5_OptModelinAtten',  'beta0.5_OptModesoft',
    # 'beta0.1_OptModeelim',    
    'beta0.1_OptModelinAtten',  'beta0.1_OptModesoft',
    'beta0.05_OptModeelim',   'beta0.05_OptModelinAtten', 'beta0.05_OptModesoft',
    'beta0.2_OptModeelim',    'beta0.2_OptModelinAtten',  'beta0.2_OptModesoft'
]

if __name__ == "__main__":
    # parallelize over subjects for each addon
    for addon in addons:
        Parallel(n_jobs=-1, verbose=5)(
            delayed(run_subject)(subject, addon)
            for subject in range(55)
        )
