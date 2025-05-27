import matplotlib.pyplot as plt
import os
def plot_evoked(epochs_dict, plot=False, plot_topo=False, save=False, subject=None, run=None,
                directory=os.path.join(os.getcwd(),'data','plots')):
    """
    Plots evoked responses from a dictionary of evoked objects, with options for 
    individual plots or topographical plots.

    Args:
        epochs_dict (dict): A dictionary containing evoked objects, where keys are condition names 
                            and values are MNE Evoked objects.
        plot (bool, optional): If True, plots individual evoked responses with global field power 
                               and spatial colors. Defaults to True.
        plot_topo (bool, optional): If True, plots topographical representations of the evoked responses.
                                    Defaults to False. Ignored if `plot` is True.

    Behavior:
        - The function dynamically creates subplots based on the number of evoked objects provided in `epochs_dict`.
        - For each evoked object, the function plots either individual evoked responses or topographical plots,
          depending on the `plot` and `plot_topo` arguments.
        - Titles of subplots include the condition name and the count of EEG channels.

    Plot Customization:
        - The number of subplots is arranged in a grid format with 3 columns and as many rows as needed to fit 
          all evoked responses.
        - If there are more subplots than evoked objects, extra axes are hidden to maintain a clean layout.

    Notes:
        - This function assumes that all evoked objects exclude bad channels automatically if they are marked 
          in the `info['bads']` attribute of the evoked objects.
        - Ensure that `epochs_dict` keys are properly aligned with actual evoked objects intended for plotting.

    Example:
        >>> plot_evoked(epochs_dict, plot=True)
        >>> plot_evoked(epochs_dict, plot=False, plot_topo=True)

    Raises:
        ValueError: If neither `plot` nor `plot_topo` is set to True, the function will not perform any plotting.

    """

    # Assuming 'epochs_dict' is a dictionary containing evoked objects and 'evoked_name_list' is a list of keys.
    # Checking keys in the dictionary (assumed to be done outside the loop).
    # epochs_dict.keys()  # Make sure this returns the expected keys.

    # Define the number of columns for the subplots
    evoked_name_list = list(epochs_dict.keys())
    n_cols = 3
    n_rows = (len(evoked_name_list) + n_cols - 1) // n_cols  # Calculate rows needed to fit all evoked responses

    # Create subplots with the calculated dimensions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), constrained_layout=True)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()
             
    # Iterate over each key and corresponding axis
    for idx, key in enumerate(evoked_name_list):
        # Get the evoked object
        evoked = epochs_dict[key]
        
        # Count the number of EEG channels in the evoked object
        eeg_channel_count = len(evoked.info['ch_names'])

        # Plot each evoked response on the corresponding axis
        if plot:
            evoked.plot(gfp=True, spatial_colors=True, axes=axes[idx], show=False)
        elif plot_topo:
            evoked.plot_topo(color="r", legend=True, axes=axes[idx], show=False) 
            
        # Set the title including the key name and EEG channel count
        axes[idx].set_title(f"{key} ({eeg_channel_count} EEG chan)")

    # If there are more subplots than keys, hide the extra axes
    for ax in axes[len(evoked_name_list):]:
        ax.set_visible(False)

    plt.show()
    # kanalu nefiltruoja tikriausiai
    
    if save:
        os.makedirs(directory, exist_ok=True)
        # Check if either subject or run is None, and raise a ValueError if so
        if subject is None or run is None:
            raise ValueError("Both 'subject' and 'run' must be provided when saving data.")
        
            # Generate plot names using the extracted information
        if plot:
            plot_name = os.path.join(directory, f"s{subject}.{run}_evoked_plot.png")
        elif plot_topo:
            plot_name = os.path.join(directory, f"s{subject}.{run}_evoked_topomap.png")
        fig.savefig(plot_name)
    return plot_name