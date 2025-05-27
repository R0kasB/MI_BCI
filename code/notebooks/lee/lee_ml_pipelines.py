import sys
sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')

# Import ml_pipelines from the file you saved earlier
from pipelines.ml_pipelines import moabb_pipelines, ml_pipelines
from evaluation import train_and_evaluate

from helper_functions import setup_logger, load_procesed_data
from helper_functions import process_mi_epochs
from datasets import Lee2019
import os
import pickle
import os
from datetime import datetime

log = setup_logger("Lee_ml_pipeline_run", log_file=os.path.join("logs","Lee_full_dataset_ml_run.log"))

dataset = Lee2019()
dataset_no = 20
paradigm = "MI"

# for subject in dataset.subjects: 
for subject in range(17,55):  
    for run in [1,2]:
        for pipelines in [moabb_pipelines,ml_pipelines]:
            for pipeline_key in pipelines.keys():
                pipeline = pipelines[pipeline_key]
                try:
                    data = load_procesed_data(dataset_no, paradigm, subject, run, include=['epochs_raw'])
                    epochs = data["epochs_raw"]
                    epochs_p = process_mi_epochs(epochs)

                    results = train_and_evaluate(epochs_p, pipeline, n_splits=5, log=log,save=True)

                    directory = os.path.join(os.getcwd(), 'data','results','pipeline_scores','lee2019','2025-01-02')
                    os.makedirs(directory, exist_ok=True)
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    file_name = os.path.join(directory, f'{pipeline_key}_result_{subject:02}-{run:02}_{current_time}.pkl')
                    with open(file_name, 'wb') as f:
                        pickle.dump(results, f)
                        log.info(f"{file_name} - saved")
                except Exception as e:
                    # Log the error and continue with the next iteration
                    log.error(f"Error processing subject {subject}, run {run}")
                    log.error(f"Error details: {e}")
                    continue
                    
        