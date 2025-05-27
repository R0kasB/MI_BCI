import sys
import os
import pickle
from datetime import datetime

sys.path.append('c:\\Users\\rokas\\Documents\\GitHub\\BCI\\mi-bci\\code')
from pipelines.ml_pipelines import all_ml_pipelines
from evaluation import train_and_evaluate
from helper_functions import setup_logger, load_procesed_data, process_mi_epochs
from datasets import Lee2019


# Define the folder for the checkpoint
CHECKPOINT_FOLDER = os.path.join("data", "checkpoints")
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

CHECKPOINT_FILE = os.path.join(CHECKPOINT_FOLDER, "checkpoint.pkl")

def save_checkpoint(subject, run, pipeline_key):
    """Save the current progress to a checkpoint file."""
    checkpoint = {"subject": subject, "run": run, "pipeline_key": pipeline_key}
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {checkpoint}")

def load_checkpoint():
    """Load progress from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    # Set up logger
    log = setup_logger("Lee_ml_pipeline_run", log_file=os.path.join("logs", "Lee_full_dataset_ml_run.log"))
    dataset_no = 20
    paradigm = "MI"

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_subject = checkpoint['subject'] if checkpoint else 1
    start_run = checkpoint['run'] if checkpoint else 1
    start_pipeline_key = checkpoint['pipeline_key'] if checkpoint else None

    try:
        for subject in range(start_subject, 55):
            if subject != start_subject:
                start_run = 1
            for run in range(start_run, 3):
                pipeline_keys = list(all_ml_pipelines.keys())
                if start_pipeline_key and subject == start_subject and run == start_run:
                    pipeline_keys = pipeline_keys[pipeline_keys.index(start_pipeline_key)+1:]
                    start_pipeline_key = None

                for pipeline_key in pipeline_keys:
                    pipeline = all_ml_pipelines[pipeline_key]
                    try:
                        data = load_procesed_data(dataset_no, paradigm, subject, run, include=['epochs_raw'])
                        epochs = data["epochs_raw"]
                        epochs_p = process_mi_epochs(epochs)

                        results = train_and_evaluate(epochs_p, pipeline, n_splits=5, log=log, save=True, 
                                                    single_pipeline_key = pipeline_key, subject=subject, run=run)

                        directory = os.path.join(os.getcwd(), 'data', 'results', 'pipeline_scores', 'lee2019', '2025-01-02')
                        os.makedirs(directory, exist_ok=True)
                        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                        file_name = os.path.join(directory, f'{pipeline_key}_result_{subject:02}-{run:02}_{current_time}.pkl')
                        with open(file_name, 'wb') as f:
                            pickle.dump(results, f)
                            log.info(f"{file_name} - saved")
                                                    
                    except Exception as e:
                        log.error(f"Error processing subject {subject}, run {run}, pipeline {pipeline_key}")
                        log.error(f"Error details: {e}")
                    
                    # Save checkpoint after processing each pipeline
                    save_checkpoint(subject, run, pipeline_key)
    except KeyboardInterrupt:
        print("Interrupted. Progress saved.")
    except Exception as e:
        print(f"An error occurred: {e}. Progress saved.")

if __name__ == "__main__":
    main()
