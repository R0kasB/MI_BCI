import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')

import os
import pickle
from datetime import datetime
import numpy as np
import os
import os
import glob
import re
import mne
from joblib import Parallel, delayed
from evaluation import cs_evaluate
from helper_functions import setup_logger, load_procesed_data
from code.pipelines.ml_pipelines import features, classifiers



# constants
dataset_no  = 20
paradigm    = "MI"
extra_mark  = "(-5_10)(8_30Hz)"
log_dir     = "logs"

baseline = (-5, 0) 
tmin= -5
tmax= 10 
freqs = np.arange(8, 30) 

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
            include=[f"epochs_raw{addon}"], extra_mark=extra_mark
        )

        d2 = load_procesed_data(
            dataset_no, paradigm, subject, run=2,
            include=[f"epochs_raw{addon}"], extra_mark=extra_mark
        )
        epochs = {
            "1": d1[f"epochs_raw{addon}"],
            "2": d2[f"epochs_raw{addon}"]
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
            save_dir=os.path.join("models", f"cs_eval{addon}"),
            train_run=1,
            test_run=2,
            return_within=True
        )

        # --- save pickle ---
        out_dir = os.path.join(
            os.getcwd(), "data", "results", "pipeline_scores",
            "lee2019", f"2025-04-25{addon}"
        )
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = os.path.join(
            out_dir,
            f"cs_eval_result{addon}_{subject:02}_{ts}.pkl"
        )
        with open(fname, "wb") as f:
            pickle.dump(results, f)
        log.info(f"[Subject {subject:02}] Saved results to {fname}")

    except Exception as e:
        log.error(f"[Subject {subject:02}] Error: {e}")

# list of addon variants
addons = [
    '', #paprastos epohos
    '_autoreject' 
    ]   

def tfr_plot(paths, addon_tfr, save_mark):
    # --- (A) parse out the existing list, as ints ---
    pattern = re.compile(r's(\d+)\.(\d+)')
    # assume `paths` is your list of filenames
    existing = []
    for p in paths:
        fn = os.path.basename(p)
        m = pattern.search(fn)
        if not m:
            continue
        subj, run = int(m.group(1)), int(m.group(2))
        existing.append((subj, run))

    # --- (B) build all possible pairs ---
    all_pairs = [(subj, run) 
                for subj in range(54)     # subjects 0 … 53
                for run  in (1, 2)]          # runs 1 and 2


    # --- (C) find the missing ones ---
    existing_set = set(existing)
    missing = [pair for pair in all_pairs if pair not in existing_set]

    # --- (D) iterate over those and call your function ---
    for subject, run in missing:
        print(f"--> will run for subject={subject}, run={run}")
        
        try:
            data = load_procesed_data(dataset_no = 20, paradigm='MI', subject=subject, run=run, 
                                        include=[f'epochs_raw{addon_tfr}'], extra_mark=extra_mark)
            epochs = data[f"epochs_raw{addon_tfr}"]
            
            tfr = epochs.compute_tfr(
                method="multitaper",
                freqs=freqs,
                n_cycles=freqs,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=2,
                n_jobs=-1,
            )
            tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
            tfr_base_path = os.path.join(os.getcwd(), 'data','tfr','20')
            os.makedirs(tfr_base_path,exist_ok=True)
            tfr_path = os.path.join(tfr_base_path, f's{subject:02}.{run:02}{extra_mark}{save_mark}_tfr_data.h5')

            tfr.save(tfr_path, overwrite=True)
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # parallelize over subjects for each addon
    for addon in addons:
        Parallel(n_jobs=-1, verbose=5)(
            delayed(run_subject)(subject, addon)
            for subject in range(55)
        )
        
    tfr_plot(paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)autore_tfr_data.h5'),
             addon_tfr = '_autoreject',
             save_mark = 'autore'
)
    tfr_plot(paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)_tfr_data.h5'),
             addon_tfr = '',
             save_mark = ''
)
    tfr_plot(paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)_beta0.1_OptModesoft_tfr_data.h5'),
             addon_tfr = '_beta0.1_OptModesoft',
             save_mark = '_beta0.1_OptModesoft'
)