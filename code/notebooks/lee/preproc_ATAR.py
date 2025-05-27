import sys
sys.path.append('c:\\Users\\rokas\\Documents\\Github\\BCI\\mi-bci\\code')
from helper_functions import preprocess_raw_atar, setup_logger
from datasets import Lee2019

log = setup_logger("Lee_preprocess")

dataset = Lee2019()
paradigm = 'MI'
for subject in range(44,54):
    for run in [1,2]: 
        try:
            log.info(f"\n ++++++++++Starting subject-{subject} , run-{run}+++++++++++++ \n")
            raw = dataset.load_one_raw_fif(subject=subject,run=run,paradigm = paradigm)
            log.info(f"\n +++++++++raw loaded (subject-{subject} , run-{run})+++++++++++++ \n")
            
            data = preprocess_raw_atar(raw=raw, 
                        event_id=dict(left_hand=2, right_hand=1),
                        tmin=-5, tmax=10,
                        preloaded=False,
                        include_raw=False,
                        save=True,
                        extra_mark='(-5_10)(8_30Hz)',
                        dataset_no=20,
                        paradigm=paradigm,
                        subject=subject,
                        run=run,
                        filter_data=True,
                        # ATAR minimal params:
                        beta=0.1,
                        OptMode='soft',
                        optimize=True,
                        log = log)
            
            log.info(f"\n ++++++++++Successfully executed preprocessing for subject-{subject} , run-{run} +++++++++++ \n")

        except Exception as e:
            log.error(f"Error with subject-{subject}, run-{run}: {e}")