import sys
sys.path.append('c:\\Users\\rokas\\Documents\\Github\\BCI\\mi-bci\\code')
from helper_functions import preprocess_raw, setup_logger, preprocess_raw_autoreject
from datasets import Lee2019

log = setup_logger("Lee_preprocess")

dataset = Lee2019()

paradigm = "MI"
tmin = -5
tmax = 10 
event_id = dict(left_hand=2, right_hand=1)
for subject in [5,6,7,8,10,25]:#range(0,54):
    for run in [1,2]: 
        try:
            log.info(f"\n ++++++++++Starting subject-{subject} , run-{run}+++++++++++++ \n")
            raw_loaded = dataset.load_one_raw_fif(subject=subject,run=run,paradigm = paradigm)
            log.info(f"\n +++++++++raw loaded (subject-{subject} , run-{run})+++++++++++++ \n")

            # raw = raw_loaded.filter(l_freq=8, h_freq=30, method='iir', iir_params=dict(order=5, ftype='butter'))
            # log.info(f"\n ++++++++++raw filtered (subject-{subject} , run-{run})++++++++++ \n")
            raw=raw_loaded

            # data = preprocess_raw(raw, event_id = event_id, tmin=tmin, tmax=tmax, dataset_no=20,
            #                    subject=subject, run=run,paradigm=paradigm, save=True, n_jobs=-1)

            data = preprocess_raw_autoreject(raw, event_id = event_id, tmin=tmin, tmax=tmax, dataset_no=20,
                                subject=subject, run=run,paradigm=paradigm, save=True, extra_mark='(-5_10)(8_30Hz)',
                                filter_data=True,preloaded=False)
            log.info(f"\n ++++++++++Successfully executed preprocessing for subject-{subject} , run-{run} +++++++++++ \n")

        except Exception as e:
            log.error(f"Error with subject-{subject}, run-{run}: {e}")