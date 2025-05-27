import sys
sys.path.append('c:\\Users\\rokas\\Documents\\Github\\BCI\\mi-bci\\code')
from helper_functions import setup_logger
import logging

import os
import glob
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import gc

import mne

extra_mark  = '(-5_10)(8_30Hz)'
tfr_dir     = r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\data\tfr\20'
plot_dir    = r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\plots\bakiui'

# ensure plot directory exists
os.makedirs(plot_dir, exist_ok=True)

def plot_tfrs(tfr_paths, plot_mark,log):
    pattern   = re.compile(r's(\d+)\.(\d+)')
    for tfr_path in tfr_paths:
        fn = os.path.basename(tfr_path)
        m = pattern.search(fn)
        if not m:
            continue
        
        try:
            subject, run = int(m.group(1)), int(m.group(2))

            # define where the plot would live
            plot_name = f's{subject:02}.{run:02}{extra_mark}autore_tfr_plot.png'
            plot_path = os.path.join(plot_dir, plot_name)

            # skip if already plotted
            if os.path.isfile(plot_path):
                log.info(f"[SKIP] plot exists for s{subject:02}.{run:02}")
                continue
            # events = dict(left_hand=2, right_hand=1)

            # load the precomputed TFR
            tfr = mne.time_frequency.read_tfrs(tfr_path)

            df = tfr.to_data_frame(time_format=None, long_format=True)
            freq_bounds = {"_": 7,"alpha": 13, "beta": 30}
            df["band"] = pd.cut(
                df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
            )
            df = df.dropna()

            df = df[df["channel"].isin(["C3", "Cz", "C4"])].copy()

            df["channel"] = df["channel"].astype("category")
            df["channel"] = df["channel"].cat.set_categories(["C3", "Cz", "C4"], ordered=True)
            
            
            # build your lowercase-series once
            conds = df['condition'].astype(str).str.lower()

            # masks
            is_right = conds.isin(['1']) | conds.str.contains('right', regex=False)
            is_left  = conds.isin(['2']) | conds.str.contains('left',  regex=False)

            # map to new strings (default keeps the original if you want, or put np.nan)
            df['condition'] = (
                pd.Series(np.where(is_right, 'Dešinė ranka',
                        np.where(is_left,  'Kairė ranka',
                                df['condition'].astype(str))))
            )

            # now convert to categorical in one go; categories will be only those actually present
            df['condition'] = df['condition'].astype('category')

            
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
            
            g.fig.suptitle(f's{subject:02}.{run:02}{extra_mark}{plot_mark}', fontsize=16)

            g.fig.subplots_adjust(top=0.88, left=0.1, right=0.9, bottom=0.08)
                        
            plt.tight_layout()

            plot_path = os.path.join(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\plots\bakiui', f's{subject:02}.{run:02}{extra_mark}{plot_mark}_tfr_plot.png')
            os.makedirs(r'C:\Users\rokas\Documents\GitHub\BCI\mi-bci\plots\bakiui', exist_ok=True)
            g.fig.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            
            plt.close()
            log.info(f"s{subject:02}.{run:02} ploted")

            # at end of plot_tfrs, after plt.close():
            del tfr, df, g
            gc.collect()
        except Exception as e:
            log.error(f"Error with subject-{subject}, run-{run}: {e}")
            
# for tfr_paths, plot_mark in zip(
#     [
#         glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}autore_tfr_data.h5')),
#         glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_tfr_data.h5')),
#         glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_beta0.1_OptModesoft_tfr_data.h5')),
#         glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_beta0.5_OptModesoft_tfr_data.h5')),
#         glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_beta0.05_OptModesoft_tfr_data.h5')),
#     ],
#     [
#         'autoreject',
#         '',
#         'ATAR_beta0.1_OptModesoft',
#         'ATAR_beta0.5_OptModesoft',
#         'ATAR_beta0.05_OptModesoft',
#     ]
# ):
#     try:
#         plot_tfrs(tfr_paths=tfr_paths, plot_mark=plot_mark, log=log)
#     except Exception as e:
#         log.error(f"Error with {plot_mark}: {e}")
        
def main():
    print("▶︎  main() has started")      # ← sanity check

    os.makedirs(plot_dir, exist_ok=True)

    # turn INFO-level messages on for root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )
    
    # make sure our helper logger inherits that
    log = setup_logger("Lee_tfr_plot")

    # build the list of jobs
    jobs = list(zip(
        [
            glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_autoreject_tfr_data.h5')),
            # glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_tfr_data.h5')),
            glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_beta0.1_OptModesoft_tfr_data.h5')),
            glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_beta0.5_OptModesoft_tfr_data.h5')),
            glob.glob(os.path.join(tfr_dir, f's*.*{extra_mark}_beta0.05_OptModesoft_tfr_data.h5')),
        ],
        [
            '_autoreject',
            # '',
            '_atar_beta0.1_OptModesoft',
            '_atar_beta0.5_OptModesoft',
            '_atar_beta0.05_OptModesoft',
        ]
    ))

    # log.info(f"Dispatching {len(jobs)}")
    # Parallel(n_jobs=1, verbose=5)(
    #     delayed(plot_tfrs)(tfr_paths, plot_mark, log)
    #     for tfr_paths, plot_mark in jobs
    # )
    for tfr_paths, plot_mark in jobs:
        plot_tfrs(tfr_paths, plot_mark, log)

    log.info("All jobs complete.")

if __name__ == "__main__":
    main()