import matplotlib.pyplot as plt
def plot_raw(raw, cleaned_raw):
    '''
    raw.plot(scalings='auto', title='Original Raw Data', show=True)
    cleaned_raw.plot(scalings='auto', title='Cleaned Raw Data', show=True)
    raw.plot_psd(fmax=50, title='Original Raw Data psd', show=True)
    cleaned_raw.plot_psd(fmax=50,title='Cleaned Raw Data psd', show=True)
    plt.show()
    '''
    # Extract data and times for plotting
    raw_data, raw_times = raw.get_data(), raw.times
    cleaned_data, cleaned_times = cleaned_raw.get_data(), cleaned_raw.times

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the original raw data (first 10 channels)
    axs[0, 0].plot(raw_times, raw_data[:10].T)
    axs[0, 0].set_title('Original Raw Data')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude (µV)')

    # Plot the cleaned raw data (first 10 channels)
    axs[0, 1].plot(cleaned_times, cleaned_data[:10].T)
    axs[0, 1].set_title('Cleaned Raw Data')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Amplitude (µV)')

    # Plot the Power Spectral Density (PSD) of the original raw data
    raw.plot_psd(fmax=50, ax=axs[1, 0], show=False)
    axs[1, 0].set_title('Original Raw Data PSD')

    # Plot the Power Spectral Density (PSD) of the cleaned raw data
    cleaned_raw.plot_psd(fmax=50, ax=axs[1, 1], show=False)
    axs[1, 1].set_title('Cleaned Raw Data PSD')

    # Add titles and adjust layout
    fig.suptitle('Comparison of Original and Cleaned EEG Data', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the main title
    plt.show()