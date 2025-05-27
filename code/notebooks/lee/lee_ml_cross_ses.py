import sys
import os
import pickle
from datetime import datetime

sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')
from evaluation import cs_evaluate
from helper_functions import setup_logger, load_procesed_data
from datasets import Lee2019
from pipelines.ml_pipelines_ba import features, classifiers
addon = ''
def main():
    # Set up logger
    log = setup_logger("Lee_ml_pipeline_run", log_file=os.path.join("logs", f"Lee_full_dataset_ml_run_cs_eval{addon}_04_20.log"))
    os.makedirs("logs", exist_ok=True)
    dataset_no = 20
    paradigm = "MI"

    for subject in range(0,55):
        try:
            data_run1 = load_procesed_data(dataset_no, paradigm, subject, run=1, include=[f'epochs_raw{addon}'], 
                                        extra_mark=('(-5_10)(8_30Hz)'))
            epochs_run1 = data_run1[f"epochs_raw{addon}"]

            data_run2 = load_procesed_data(dataset_no, paradigm, subject, run=2, include=[f'epochs_raw{addon}'], 
                                        extra_mark=('(-5_10)(8_30Hz)'))
            epochs_run2 = data_run2[f"epochs_raw{addon}"]
            
            epochs = {'1':epochs_run1,'2':epochs_run2}

            results = cs_evaluate(epochs, n_splits=5, log=log, save_model=True, subject=subject,
                                    construct_pipelines=True,feature_pipelines=features,
                                    classifier_pipelines=classifiers,n_jobs=-1,
                                    save_dir = os.path.join('models',f'cs_eval{addon}'),
                                    train_run=1,test_run=2)

            directory = os.path.join(os.getcwd(), 'data', 'results', 'pipeline_scores', 'lee2019', f'2025-04-20{addon}')
            os.makedirs(directory, exist_ok=True)
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            file_name = os.path.join(directory, f'cs_eval_result{addon}_{subject:02}_{current_time}.pkl')
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)
                log.info(f"{file_name} - saved")
                                        
        except Exception as e:
            log.error(f"Error processing subject {subject}")
            log.error(f"Error details: {e}")
            
if __name__ == "__main__":
    main()
