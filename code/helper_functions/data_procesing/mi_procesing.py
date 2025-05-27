def process_mi_epochs(epochs):
    """
    Picks MI-related channels and band-pass filters them between 8-30 Hz.
    """
    channels_to_pick = [
        'FC5','FC3','FC1','FC2','FC4','FC6',
        'C5','C3','C1','Cz','C2','C4','C6',
        'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
    ]
    picks = [ch for ch in channels_to_pick if ch in epochs.ch_names]
    if not picks:
        print("No MI channels found; returning original epochs.")
        return epochs
    # use inst.pick(...) instead of deprecated pick_channels
    epochs_proc = epochs.copy().pick(picks)
    epochs_proc = epochs_proc.filter(
        l_freq=8, h_freq=30,
        method='iir',
        iir_params=dict(order=5, ftype='butter'),
        n_jobs=1
    )
    return epochs_proc


def process_mi_raw(raw):
    """
    Picks MI-related channels and band-pass filters raw data between 8-30 Hz.
    """
    channels_to_pick = [
        'FC5','FC3','FC1','FC2','FC4','FC6',
        'C5','C3','C1','Cz','C2','C4','C6',
        'CP5','CP3','CP1','CPz','CP2','CP4','CP6'
    ]
    picks = [ch for ch in channels_to_pick if ch in raw.ch_names]
    if not picks:
        print("No MI channels found; returning original raw.")
        return raw
    raw_proc = raw.copy().pick(picks)
    raw_proc = raw_proc.filter(
        l_freq=8, h_freq=30,
        method='iir',
        iir_params=dict(order=5, ftype='butter'),
        n_jobs=1
    )
    return raw_proc