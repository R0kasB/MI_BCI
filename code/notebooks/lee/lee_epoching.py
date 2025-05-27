import sys
import mne
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')
from helper_functions import preprocess_raw_autoreject, save_epochs, setup_logger
from datasets import Lee2019
from helper_functions.data_procesing.mi_procesing import process_mi_raw

log = setup_logger("Lee_preprocess")

dataset = Lee2019()

# dataset.initialize()

paradigm = "MI"
tmin = -5
tmax = 10
#epohuoja tik su autorejectu

for subject in range(0,54):
    for run in [1,2]:
        try:
            raw_temp = dataset.load_one_raw_fif(subject=subject,paradigm=paradigm,run=run)
            raw = process_mi_raw(raw_temp)
            event_id = dict(left_hand=2, right_hand=1)
            
            # events_raw = mne.find_events(raw)
            events_raw, event_id = mne.events_from_annotations(raw)

            epochs_raw = mne.Epochs(raw, events_raw, event_id=event_id, tmin=tmin, tmax=tmax)
            
            #epochs_raw, epochs_raw_autoreject = preprocess_raw_autoreject(raw, event_id, preloaded=False, include_raw=True, tmin=tmin,tmax=tmax) 
            
            log.info(f"Preprocesed, subject-{subject} , run-{run} ")
            
            dict_to_save = {
                'epochs_raw': epochs_raw,
                #'epochs_raw_autoreject': epochs_raw_autoreject,
            }

            save_epochs(dict_to_save, dataset_no='20', subject=subject, run=run, paradigm=paradigm, extra_mark='(-5_10)(8_30Hz)')
            log.info(f"Saved, subject-{subject} , run-{run} ")
            
        except Exception as e:
            log.error(f"Error with subject-{subject}, run-{run}: {e}")
            
            