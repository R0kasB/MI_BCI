import mne
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
import numpy as np
import os

def plot_epochs_image(epochs, save=True ,directory= os.path.join(os.getcwd(),'data','plots')):
    """
    Visualizes EEG epochs data by generating and saving plots for different conditions 
    ('left hand', 'right hand', 'passive state').

    Args:
        epochs (mne.Epochs): The MNE epochs object containing the EEG data.
        save (bool, optional): Whether to save the plots as PNG files. Defaults to True.
        directory (str): The directory where plots will be saved. Defaults to a 'plots' 
                         folder in the current working directory.

    Raises:
        ValueError: If no filenames are associated with the epochs object.
    # plot_epoch_image(epochs, directory= os.path.join(os.getcwd(),'data','plots')) 
    by chatgpt
    """
    os.makedirs(directory, exist_ok=True)
    if epochs.filename:
        epochs_filename = os.path.basename(epochs.filename)
    else:
        raise ValueError("No filenames associated with the epochs object.")

    # Extract subject and run information from the filename
    filename_parts = epochs_filename.split('_')
    subject_run_part = filename_parts[0]  # e.g., 's1.0'
    epochs_name = '_'.join(filename_parts[1:]).split('-')[0]  # Extracts the condition name

    # Generate plot names using the extracted information
    plot_name = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_epoch_image_unfiltered.png")
    plot_name_motor_ch = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_epoch_image_motor_ch.png") 
    plot_name_motor_ch_alpha = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_epoch_image_motor_ch_alpha_band.png") 

    def _plot(epochs_filtered, plot_title, save_name):
        """
        Internal function to handle plotting of epochs images for specific conditions.
        
        Args:
            epochs_filtered (mne.Epochs): Filtered epochs data to be plotted.
            plot_title (str): The title of the plot.
            save_name (str): The filename to save the plot.
        """
        # Get the unique event names (conditions) from the epochs
        conditions = list(epochs_filtered.event_id.keys())
        n_conditions = len(conditions)

        # Set up the figure and grid spec based on the number of conditions
        fig = plt.figure(figsize=(5 * n_conditions, 6), constrained_layout=True)
        fig.suptitle(plot_title, fontsize=16)
        
        # Create a GridSpec for the plots
        gs = GridSpec(2, 2 * n_conditions, figure=fig, width_ratios=[2.4, 0.1] * n_conditions)

        # Loop over the conditions and plot
        for idx, condition in enumerate(conditions):
            col_start = idx * 2
            ax_main = fig.add_subplot(gs[0, col_start])
            ax_colorbar = fig.add_subplot(gs[0, col_start + 1])
            ax_evoked = fig.add_subplot(gs[1, col_start:col_start + 2])

            # Plot the epochs image for the current condition
            mne.viz.plot_epochs_image(epochs_filtered[condition], picks='eeg',
                                    axes=[ax_main, ax_evoked, ax_colorbar], show=False)
            ax_main.set_title(condition.replace('_', ' ').title())

        # Save the figure if save_name is provided
        if save_name:
            plt.savefig(save_name)
        plt.show()

    # Define relevant channels for motor area
    relevant_channels = [
    'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
    ]

    # Ensure these channels exist in the epochs data
    existing_motor_channels = [ch for ch in relevant_channels if ch in epochs["right_hand"].ch_names]

    _plot(epochs,'All freq, all chanels',save_name = plot_name)
    if not existing_motor_channels:
        print("No motor area channels found in the epochs data. Check your channel names.")
    else:
        epochs_filtered = epochs.copy().pick_channels(existing_motor_channels)
        _plot(epochs_filtered,f'Motor Area Channels Only ({existing_motor_channels})', plot_name_motor_ch)

        low_freq = 8  # Lower bound of the frequency range
        high_freq = 12  # Upper bound of the frequency range

    # Filter the epochs to include only the specified frequency range
        epochs_alpha_filtered = epochs.copy().filter(l_freq=low_freq, h_freq=high_freq)
        epochs_alpha_filtered = epochs_alpha_filtered.pick_channels(existing_motor_channels)
        _plot(epochs_alpha_filtered,f'Motor Area Channels Only - Alpha Band ({low_freq}-{high_freq}Hz)', plot_name_motor_ch_alpha)
# plot_epoch_image(epochs, directory= os.path.join(os.getcwd(),'data','plots')) 
    

def _plot_tfr(epochs, conditions, power_list, directory, vmin, vmax, save):
    """
    Helper function to plot Time-Frequency Representations (TFR) for given conditions.
    
    Args:
        epochs (mne.Epochs): The MNE epochs object containing the data.
        conditions (list): List of conditions to plot.
        power_list (list): List of computed TFR power objects for each condition.
        directory (str): Directory to save the plots.
        vmin (float): Minimum value for color scale.
        vmax (float): Maximum value for color scale.
        by chatgpt
    """
    def _plot(conditions_plot, save_path, title, power_list, y_limits,save):
        """
        Internal function to create individual TFR plots.

        Args:
            conditions_plot (tuple): Conditions and color limits for the plot.
            save_path (str): File path to save the plot.
            title (str): Title of the plot.
            power_list (list): List of TFR power objects.
            y_limits (tuple): Y-axis limits for the frequency range.
            by chatgpt
        """
        conditions, vmin, vmax = conditions_plot
        fontsize = 14
        ticksize = 12
        
        # Create subplots for each condition
        fig, axes = plt.subplots(len(conditions), 1, figsize=(18, 12), constrained_layout=True)

        # Ensure axes is always iterable
        if len(conditions) == 1:
            axes = [axes]

        # Plot TFR for each condition
        for ic, (condition, power) in enumerate(zip(conditions, power_list)):
            power.plot([0], axes=axes[ic], cmap='viridis', vlim=[vmin, vmax], show=False)
            axes[ic].set_title(f'{condition} Baseline Corrected', fontsize=fontsize)
            axes[ic].set_ylim(y_limits)
            axes[ic].tick_params(axis='x', labelsize=ticksize)  # Time labels
            axes[ic].tick_params(axis='y', labelsize=ticksize)  # Frequency (Hz) labels
            axes[ic].xaxis.label.set_size(fontsize)  # X-axis label font size
            axes[ic].yaxis.label.set_size(fontsize)  # Y-axis label font size

        fig.suptitle(title, fontsize=fontsize)
        if save:
            fig.savefig(save_path)
        plt.show()

    # Extract filename from epochs object for naming plots
    if epochs.filename:
        epochs_filename = os.path.basename(epochs.filename)
    else:
        raise ValueError("No filenames associated with the epochs object.")
    
    # Extract subject and run information from the filename
    filename_parts = epochs_filename.split('_')
    subject_run_part = filename_parts[0]  # e.g., 's1.0'
    epochs_name = '_'.join(filename_parts[1:]).split('-')[0]  # Extracts the condition name

    # Generate plot names using the extracted information
    plot_name_all = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_all_channels_TFR.png")
    plot_name_selected = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_selected_channels_TFR.png") 
    
    # Define relevant channels for Motor Imagery (MI) BCI
    relevant_channels = ['C3', 'C4', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2', 'Pz']
    present_channels = [ch for ch in relevant_channels if ch in power_list[0].info['ch_names']]

    # Plot for all channels
    _plot((conditions, vmin, vmax), save_path=plot_name_all, title='Time-Frequency Representation for All Channels',
        power_list=power_list, y_limits=(0, 50),save=save)

    # Plot for selected relevant channels if available
    if present_channels:
        power_list_selected = [power.copy().pick_channels(present_channels) for power in power_list]
        _plot((conditions, vmin, vmax), save_path=plot_name_selected,
            title=f'Time-Frequency Representation for Channels {", ".join(present_channels)}',
            power_list=power_list_selected, y_limits=(4, 18),save=save)

def plot_epochs_TFR(epochs, save=True ,directory= os.path.join(os.getcwd(),'data','plots')):
    """
    Plots the Time-Frequency Representation (TFR) of EEG epochs.
    
    Args:
        epochs (mne.Epochs): The MNE epochs object containing the data.
        save (bool, optional): Whether to save the plots as PNG files. Defaults to True.        
        directory (str): Directory where the plots will be saved.
        
    by chatgpt
    """
    os.makedirs(directory, exist_ok=True)
    
    # Extract conditions from the epochs object
    conditions = list(epochs.event_id.keys())

    # Define frequencies of interest and the number of cycles
    freqs = np.linspace(1, 50, 100)  # Frequencies from 1 Hz to 50 Hz
    n_cycles = freqs / 4  # Number of cycles per frequency

    # Compute TFRs and apply baseline correction    
    power_list = []
    for condition in conditions:
        epochs_condition = epochs[condition]
        power = epochs_condition.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles, 
                                            use_fft=True, average=True, decim=3, n_jobs=-1)
        power.apply_baseline(baseline=(-0.2, 0), mode='zscore')
        power_list.append(power)

    # Determine global vmin and vmax for plotting
    all_data = np.concatenate([p.data for p in power_list], axis=0)
    vmin = np.percentile(all_data, 5)  # 5nd percentile for vmin
    vmax = np.percentile(all_data, 95) # 95nd percentile for vmax

    # Call helper function to plot TFRs
    _plot_tfr(epochs, conditions, power_list, directory, vmin, vmax, save)
    
# Call the function with the required parameters
# plot_TFR(epochs, 'data/plots')

def plot_epochs_topomap(epochs,save=True ,directory= os.path.join(os.getcwd(),'data','plots','topomap')):
    """
    Visualizes Power Spectral Density (PSD) topomaps for different conditions within EEG epochs data.

    This function generates and saves topomap plots of the EEG data for various conditions 
    found in the epochs object. It produces two sets of plots: one with non-normalized PSD data 
    and another with normalized PSD data, illustrating the power distribution across the scalp.

    Args:
        epochs (mne.Epochs): The MNE epochs object containing the EEG data, typically preprocessed.
        save (bool, optional): Whether to save the plots as PNG files. Defaults to True.
        directory (str, optional): The directory where the generated plots will be saved. Defaults 
                                   to a 'topomap' folder within a 'plots' folder in the current 
                                   working directory.

    Raises:
        ValueError: If no filenames are associated with the epochs object.

    The function performs the following steps:
    1. Ensures the specified directory exists, creating it if necessary.
    2. Extracts subject and condition information from the filename associated with the epochs object.
    3. Generates filenames for saving the plots based on extracted information.
    4. Creates non-normalized PSD topomap plots:
       - Computes the PSD for each condition and plots the topomap without normalization.
       - Each row in the subplot corresponds to a different condition, with columns for different 
         frequency bands.
       - Adds labels to each row indicating the condition.
       - The figure is titled and saved as a PNG file in the specified directory.
    5. Creates normalized PSD topomap plots:
       - Repeats the plotting process with normalized PSD data.
       - Saves the normalized plots with a descriptive title in the specified directory.

    The resulting topomaps provide a visual representation of the power distribution across the scalp 
    for each condition, aiding in the analysis of neural patterns associated with different tasks 
    or states in EEG studies.
    
    by chatgpt
    """
    os.makedirs(directory, exist_ok=True)
    if epochs.filename:
        epochs_filename = os.path.basename(epochs.filename)
    else:
        raise ValueError("No filenames associated with the epochs object.")
    
    # Extract subject and run information from the filename
    filename_parts = epochs_filename.split('_')
    subject_run_part = filename_parts[0]  # e.g., 's1.0'
    epochs_name = '_'.join(filename_parts[1:]).split('-')[0]  # Extracts the condition name

    # Generate plot names using the extracted information
    plot_name_not_normalized = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_epochs_topomap_not_normalized.png")
    plot_name_normalized = os.path.join(directory, f"{subject_run_part}_{epochs_name}-_epochs_topomap_normalized.png") 

    conditions = list(epochs.event_id.keys())
    fig, ax = plt.subplots(len(conditions), 5, figsize=(16, 8), constrained_layout=True)
    for index, condition in enumerate(conditions):
        epochs[condition].compute_psd().plot_topomap(normalize=False, axes=ax[index,:], show=False)
        ax[index, 0].set_ylabel(f'{condition}', fontsize=16, labelpad=20)

    fig.suptitle('Power Spectral Density Topomaps, NOT normalized')
    if save:
        fig.savefig(plot_name_not_normalized)
    plt.show()
    

    fig2, ax2 = plt.subplots(len(conditions), 5, figsize=(16, 8), constrained_layout=True)
    for index, condition in enumerate(conditions):
        epochs[condition].compute_psd().plot_topomap(normalize=True, axes=ax2[index,:], show=False)
        # Place a row title above the middle of each row using fig2.text
        ax2[index, 0].set_ylabel(f'{condition}', fontsize=16, labelpad=20)
    fig2.suptitle('Power Spectral Density Topomaps, Normalized')
    if save:
        fig2.savefig(plot_name_normalized)
    plt.show()
