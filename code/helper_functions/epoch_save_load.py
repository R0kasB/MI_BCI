import mne
import os

def save_epochs(data_to_save: dict, dataset_no: str, paradigm:str, subject, run, extra_mark):
    ''' Data to save available keys : epochs_raw, epochs_raw_cleaned,epochs_raw_autoreject, epochs_raw_cleaned_autoreject, raw_cleaned, raw'''
    root_path = os.path.join(os.getcwd(),'data','procesed',dataset_no, paradigm,extra_mark,str(subject),str(run))
    os.makedirs(root_path, exist_ok=True)
    # [epochs_raw, epochs_raw_cleaned,epochs_raw_autoreject, epochs_raw_cleaned_autoreject, raw_cleaned, raw]
    if "raw" in data_to_save.keys():
        data_to_save["raw"].save(os.path.join(root_path, 's{:02}_{:02}_raw-raw.fif'.format(subject, run)), overwrite=True)
    if "raw_cleaned" in data_to_save.keys():
        data_to_save["raw_cleaned"].save(os.path.join(root_path, 's{:02}.{:02}_raw_cleaned-raw.fif'.format(subject, run)), overwrite=True)
    if "epochs_raw" in data_to_save.keys():
        data_to_save["epochs_raw"].save(os.path.join(root_path, 's{:02}.{:02}_epochs_raw-epo.fif'.format(subject, run)), overwrite=True)
    if "epochs_raw_cleaned" in data_to_save.keys():
        data_to_save["epochs_raw_cleaned"].save(os.path.join(root_path, 's{:02}.{:02}_epochs_raw_cleaned-epo.fif'.format(subject, run)), overwrite=True)
    if "epochs_raw_autoreject" in data_to_save.keys():
        data_to_save["epochs_raw_autoreject"].save(os.path.join(root_path, 's{:02}.{:02}_epochs_raw_autoreject-epo.fif'.format(subject, run)), overwrite=True)
    if "epochs_raw_cleaned_autoreject" in data_to_save.keys():
        data_to_save["epochs_raw_cleaned_autoreject"].save(os.path.join(root_path, 's{:02}.{:02}_epochs_raw_cleaned_autoreject-epo.fif'.format(subject, run)), overwrite=True)
        
def load_procesed_data(dataset_no, paradigm, subject, run, 
                       include=['epochs_raw', 'epochs_raw_cleaned', 'epochs_raw_autoreject', 'epochs_raw_cleaned_autoreject', 'raw_cleaned', 'raw'],
                       extra_mark=''):
    '''raw, raw_cleaned, epochs_raw, epochs_raw_cleaned, epochs_raw_autoreject, epochs_raw_cleaned_autoreject'''
    root_path = os.path.join(os.getcwd(),'data','procesed',str(dataset_no), paradigm,extra_mark,str(subject),str(run))
    return_data = {}
    if 'raw' in include:
        raw_path = os.path.join(os.getcwd(),'data','raw_fif',str(dataset_no), paradigm)
        raw = mne.io.read_raw_fif(os.path.join(raw_path, 's{:02}.{:02}_raw.fif'.format(subject, run)), preload=True)
        return_data['raw'] = raw
    if 'raw_cleaned' in include:
        raw_cleaned = mne.io.read_raw_fif(os.path.join(root_path, 's{:02}.{:02}_raw_cleaned-raw.fif'.format(subject, run)), preload=True)
        return_data['raw_cleaned'] = raw_cleaned
    if 'epochs_raw' in include:
        epochs_raw = mne.read_epochs(os.path.join(root_path, 's{:02}.{:02}_epochs_raw-epo.fif'.format(subject, run)),preload=True)
        return_data['epochs_raw'] = epochs_raw
    if 'epochs_raw_cleaned' in include:
        epochs_raw_cleaned = mne.read_epochs(os.path.join(root_path, 's{:02}.{:02}_epochs_raw_cleaned-epo.fif'.format(subject, run)),preload=True)
        return_data['epochs_raw_cleaned'] = epochs_raw_cleaned
    if 'epochs_raw_autoreject' in include:
        epochs_raw_autoreject = mne.read_epochs(os.path.join(root_path, 's{:02}.{:02}_epochs_raw_autoreject-epo.fif'.format(subject, run)),preload=True)
        return_data['epochs_raw_autoreject'] = epochs_raw_autoreject
    if 'epochs_raw_cleaned_autoreject' in include:
        epochs_raw_cleaned_autoreject = mne.read_epochs(os.path.join(root_path, 's{:02}.{:02}_epochs_raw_cleaned_autoreject-epo.fif'.format(subject, run)),preload=True)
        return_data['epochs_raw_cleaned_autoreject'] = epochs_raw_cleaned_autoreject

    for key in include:
        if key.startswith('epochs_atar_'):
            # drop the "epochs_atar_" prefix to get the filename suffix
            suffix = key.replace('epochs_atar_', '')
            # build e.g. "s01.01_beta0.1_OptModeelim-epo.fif"
            fn = f"s{subject:02d}.{run:02d}_{suffix}-epo.fif"
            path = os.path.join(root_path, fn)
            # this will throw if the file really isn't there
            return_data[key] = mne.read_epochs(path, preload=True)
            
    return return_data

def epochs_to_evoked(raws):
    """
    Converts various epochs objects into averaged evoked responses for different conditions 
    ('right hand', 'left hand', 'passive state').

    Args:
        raws (tuple): A tuple containing epochs and raw objects, specifically:
                      (epochs_raw, epochs_raw_cleaned, epochs_raw_autoreject, 
                       epochs_raw_cleaned_autoreject, raw_cleaned, raw).
    
    Returns:
        dict: A dictionary containing evoked responses for each condition across different epochs objects.
    """
    
    #pasikeitė raw struktūra load_procesed_data fun
    
    # Unpack the raws tuple
    epochs_raw, epochs_raw_cleaned, epochs_raw_autoreject, epochs_raw_cleaned_autoreject, raw_cleaned, raw = raws

    # Initialize a dictionary to store evoked responses
    dict_evoked = {}

    # Create evoked responses for each condition and store them in the dictionary
    dict_evoked['evoked_epochs_raw_RH'] = epochs_raw["right_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_LH'] = epochs_raw["left_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_PS'] = epochs_raw["passive_state"].average(picks=['eeg'])

    dict_evoked['evoked_epochs_raw_cleaned_RH'] = epochs_raw_cleaned["right_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_cleaned_LH'] = epochs_raw_cleaned["left_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_cleaned_PS'] = epochs_raw_cleaned["passive_state"].average(picks=['eeg'])

    dict_evoked['evoked_epochs_raw_autoreject_RH'] = epochs_raw_autoreject["right_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_autoreject_LH'] = epochs_raw_autoreject["left_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_autoreject_PS'] = epochs_raw_autoreject["passive_state"].average(picks=['eeg'])

    dict_evoked['evoked_epochs_raw_cleaned_autoreject_RH'] = epochs_raw_cleaned_autoreject["right_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_cleaned_autoreject_LH'] = epochs_raw_cleaned_autoreject["left_hand"].average(picks=['eeg'])
    dict_evoked['evoked_epochs_raw_cleaned_autoreject_PS'] = epochs_raw_cleaned_autoreject["passive_state"].average(picks=['eeg'])
    
    # Return the dictionary of evoked responses
    return dict_evoked
