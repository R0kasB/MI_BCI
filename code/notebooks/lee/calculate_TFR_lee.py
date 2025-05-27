import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os
import glob
import re

from helper_functions import load_procesed_data

import mne

baseline=(-5, 0) 
tmin= -5
tmax= 10 
freqs = np.arange(8, 30) 
extra_mark = '(-5_10)(8_30Hz)'

paths = glob.glob(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20\**(-5_10)(8_30Hz)autore_tfr_data.h5')
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
             for subj in range(0, 54)     # subjects 0 … 53
             for run  in (1, 2)]          # runs 1 and 2

# --- (C) find the missing ones ---
existing_set = set(existing)
missing = [pair for pair in all_pairs if pair not in existing_set]

# --- (D) iterate over those and call your function ---
for subject, run in missing:
    print(f"--> will run for subject={subject}, run={run}")

#for subject in range(0,54):
#   for run in [1,2]:

    try:
        data = load_procesed_data(dataset_no = 20, paradigm='MI', subject=subject, run=run, 
                                    include=['epochs_raw_autoreject'], extra_mark=extra_mark)
        epochs = data["epochs_raw_autoreject"]
        
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
        tfr_path = os.path.join(tfr_base_path, f's{subject:02}.{run:02}{extra_mark}autore_tfr_data.h5')
        tfr.save(tfr_path, overwrite=True)
        
        df = tfr.to_data_frame(time_format=None, long_format=True)
        freq_bounds = {"_": 7,"alpha": 13, "beta": 30}
        df["band"] = pd.cut(
            df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
        )
        df = df.dropna()

        df = df[df["channel"].isin(["C3", "Cz", "C4"])].copy()

        df["channel"] = df["channel"].astype("category")
        df["channel"] = df["channel"].cat.set_categories(["C3", "Cz", "C4"], ordered=True)

        df['condition'] = df['condition'].replace({'left_hand': 'Kairė ranka', 'right_hand': 'Dešinė ranka'})
        g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
        g.map(sns.lineplot, "time", "value", "condition", n_boot=10)

        axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
        g.map(plt.axhline, y=0, **axline_kw)
        g.map(plt.axvline, x=0, **axline_kw)

        g.set(ylim=(None, 1.5))
        g.set_axis_labels("Laikas (s)", "ERD")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        g.add_legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.05))

        g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
        
        g.fig.suptitle(f's{subject:02}.{run:02}{extra_mark}autoreject', fontsize=16)

        g.fig.subplots_adjust(top=0.88, left=0.1, right=0.9, bottom=0.08)
                    
        plt.tight_layout()
        
        plot_path = os.path.join(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\plots\bakiui', f's{subject:02}.{run:02}{extra_mark}_tfr_plot.png')
        os.makedirs(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\plots\bakiui', exist_ok=True)
        g.fig.savefig(plot_path, dpi=300)  # You can adjust the dpi as needed
        
        plt.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")
