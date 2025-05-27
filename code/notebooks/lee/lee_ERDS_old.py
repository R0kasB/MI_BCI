import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import os
import mne
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import glob
from helper_functions import combine_images,add_border_and_title
from helper_functions import setup_logger, load_procesed_data
from datasets import Lee2019

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
    epochs_proc = epochs.copy().filter(l_freq=8, h_freq=30, method='iir', iir_params=dict(order=5, ftype='butter'))
    return epochs_proc

def plot_erds(epochs,subject,run,extra_mark=None, vmin=-1, vmax=1.5, baseline=(-1, 0), tmin=-1.0, tmax=5.0, 
              kwargs=None, save=False,plot=True, verbose=True, save_tfr=True):
    
    epochs.pick_channels(['C3','Cz','C4'])

    if not verbose:
        mne.set_log_level('ERROR')

    if kwargs is None:
        kwargs = dict(
            n_permutations=100, 
            step_down_p=0.05, 
            seed=1, 
            buffer_size=None, 
            out_type="mask"
        )
    
    freqs = np.arange(2, 36)  # frequencies from 2-35Hz
    event_ids = epochs.event_id  # {'left_hand': 2, 'right_hand': 1}
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    # Compute time-frequency representation (No changes)
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
    
    if extra_mark == None:
        extra_mark = ''
    elif extra_mark != None:
        extra_mark = extra_mark + "_"
    
    if save_tfr:
        tfr_base_path = os.path.join(os.getcwd(), 'data','tfr','20', f'{subject:02}')
        os.makedirs(tfr_base_path,exist_ok=True)
        tfr_path = os.path.join(tfr_base_path, f's{subject:02}.{run:02}_{extra_mark}tfr_data.h5')
        tfr.save(tfr_path, overwrite=True)

    # Loop over events and plot the ERDS
    for event in event_ids:
        # Select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
        )
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # Positive clusters
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
            # Negative clusters
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

            # Combine positive and negative clusters
            c = np.stack(c1 + c2, axis=2)  # Combined clusters
            p = np.concatenate((p1, p2))  # Combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # Plot TFR (ERDS map with masking)
            tfr_ev.average().plot(
                [ch],
                cmap="RdBu",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style="mask",
            )

            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # Event marker
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        
        # Colorbar and title
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event})")
        plt.tight_layout()
        
        if plot:
            plt.show()
        
        if save:
            path = os.path.join(os.getcwd(), 'data', 'plots', 'erds', f'{subject:02}')
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, f's{subject:02}.{run:02}-{event}-ERDS.png')
            plt.savefig(file_path)
            
    if save:
        if extra_mark == None:
            extra_mark = ''
        elif extra_mark != None:
            extra_mark = extra_mark + "_"
        image_paths = glob.glob(os.path.join(path, f's{subject:02}.{run:02}-**-ERDS.png'))
        combined_save_path = os.path.join(path, f's{subject:02}.{run:02}_{extra_mark}ERDS.png')
        
        combine_images(image_paths, combined_save_path, 'vertical')
        add_border_and_title(combined_save_path, combined_save_path, f's{subject:02}.{run:02}_{extra_mark}',
                             border_size=0,title_color='black',title_bg_color='white')
        
    plt.close()
    return tfr

def tfr_lineplot(tfr, show=True, save=False, extra_mark=None):
    """
    Plots the TFR data as a line plot for different frequency bands across channels.

    Args:
        tfr (mne.time_frequency.EpochsTFR): The TFR object.
        show (bool, optional): Whether to display the plot. Defaults to True.
        save (bool, optional): Whether to save the plot to a file. Defaults to False.
        save_path (str, optional): File path to save the plot if `save=True`. Defaults to None.
    """
    # Convert TFR to DataFrame in long format
    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Define frequency bands
    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
    
    # Map frequencies to bands
    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
    )

    # Filter relevant frequency bands
    freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels
    df["channel"] = df["channel"].cat.reorder_categories(["C3", "Cz", "C4"], ordered=True)

    # Create the FacetGrid plot
    g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
    g.map(sns.lineplot, "time", "value", "condition", n_boot=10)

    # Add vertical and horizontal reference lines
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)

    # Set plot limits and labels
    g.set(ylim=(None, 1.5))
    g.set_axis_labels("Time (s)", "ERDS")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Add a legend
    g.add_legend(ncol=2, loc="lower center")
    
    # Adjust the subplot layout
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    
    # Save the plot if requested
    if save:
        if extra_mark == None:
            extra_mark = ''
        elif extra_mark != None:
            extra_mark = extra_mark + "_"
            
        path = os.path.join(os.getcwd(), 'data', 'plots', 'erds', f'{subject:02}')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f's{subject:02}.{run:02}_{extra_mark}ERDS_lineplot.png')
        g.savefig(save_path)
        add_border_and_title(save_path, save_path, f's{subject:02}.{run:02}_{extra_mark}',
                             border_size=0,title_color='black',title_bg_color='white')
        
    # Show the plot if requested
    if show:
        plt.show()

    # Close the plot to avoid memory issues with multiple plots
    plt.close()
    
def tfr_violin(tfr, show=True, save=False, extra_mark=None):
    df = tfr.to_data_frame(time_format=None, long_format=True)
    # Define frequency bands
    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
    
    # Map frequencies to bands
    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
    )

    # Filter relevant frequency bands
    freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels
    df["channel"] = df["channel"].cat.reorder_categories(["C3", "Cz", "C4"], ordered=True)
    df_mean = (
        df.query("time > 1")
        .groupby(["condition", "epoch", "band", "channel"], observed=False)[["value"]]
        .mean()
        .reset_index()
    )

    g = sns.FacetGrid(
        df_mean, col="condition", col_order=["left_hand", "right_hand"], margin_titles=True
    )
    g = g.map(
        sns.violinplot,
        "channel",
        "value",
        "band",
        cut=0,
        palette="deep",
        order=["C3", "Cz", "C4"],
        hue_order=freq_bands_of_interest,
        linewidth=0.5,
    ).add_legend(ncol=4, loc="lower center")
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)

    g.map(plt.axhline, **axline_kw)
    g.set_axis_labels("", "ERDS")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    
    if save:
        if extra_mark == None:
            extra_mark = ''
        elif extra_mark != None:
            extra_mark = extra_mark + "_"
            
        path = os.path.join(os.getcwd(), 'data', 'plots', 'erds', f'{subject:02}')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f's{subject:02}.{run:02}_{extra_mark}ERDS_violin.png')
        g.savefig(save_path)
        add_border_and_title(save_path, save_path, f's{subject:02}.{run:02}_{extra_mark}',
                             border_size=0,title_color='black',title_bg_color='white')
        
    # Show the plot if requested
    if show:
        plt.show()

    # Close the plot to avoid memory issues with multiple plots
    plt.close()


log = setup_logger("Lee_preprocess")
log.info("Begining analysis")

dataset = Lee2019()

dataset_no = 20
paradigm = "MI"

for subject in dataset.subjects:
    for run in [1,2]:
        try:
            data = load_procesed_data(dataset_no, paradigm, subject, run, include=['epochs_raw', 'epochs_raw_autoreject'])
                        
            epochs = data["epochs_raw"]
            epochs_autoreject = data["epochs_raw_autoreject"]
            log.info(f"Data for s{subject:02}.{run:02} loaded")


            epochs_p = proc_epoch(epochs)
            epochs_autoreject_p = proc_epoch(epochs_autoreject)
            log.info(f"Data for s{subject:02}.{run:02} preprocessed")


            tfr = plot_erds(epochs_p,subject,run,'epochs',plot=False,save=True)
            tfr_2 = plot_erds(epochs_autoreject_p,subject,run,'autoreject',plot=False,save=True)
            log.info(f"s{subject:02}.{run:02} tfr calculated, ERDS plots plotted and saved")


            tfr_lineplot(tfr, show=False,save=True,extra_mark='epochs')
            tfr_lineplot(tfr_2, show=False,save=True,extra_mark='autoreject')
            log.info(f"s{subject:02}.{run:02} tfr lineplots plotted")


            tfr_violin(tfr, show=False,save=True,extra_mark='epochs')
            tfr_violin(tfr_2, show=False,save=True,extra_mark='autoreject')
            log.info(f"s{subject:02}.{run:02} tfr violin plots plotted")
            
            plt.close()
            log.info(f"ERDS analysis for s{subject:02}.{run:02} complete")
            
        except Exception as e:
            # Log the error and continue with the next iteration
            log.error(f"Error processing subject {subject}, run {run}")
            log.error(f"Error details: {e}")
            continue