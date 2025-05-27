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
from pipelines.ml_pipelines_ba import features, classifiers



# constants
dataset_no  = 20
paradigm    = "MI"
extra_mark  = "(-5_10)(8_30Hz)"
log_dir     = "logs"

baseline = (-5, 0) 
tmin= -5
tmax= 10 
freqs = np.arange(8, 30) 


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
                                        include=[f'epochs_atar{addon_tfr}'], extra_mark=extra_mark)
            epochs = data[f"epochs_atar{addon_tfr}"]
            
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
        
    tfr_plot(paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)_beta0.1_OptModesoft_tfr_data.h5'),
             addon_tfr = '_beta0.1_OptModesoft',
             save_mark = '_beta0.1_OptModesoft'
)
    tfr_plot(paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)_beta0.05_OptModesoft_tfr_data.h5'),
             addon_tfr = '_beta0.05_OptModesoft',
             save_mark = '_beta0.05_OptModesoft'
)
    tfr_plot(paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)_beta0.5_OptModesoft_tfr_data.h5'),
             addon_tfr = '_beta0.5_OptModesoft',
             save_mark = '_beta0.5_OptModesoft'
)