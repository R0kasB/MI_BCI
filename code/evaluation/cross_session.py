import sys
sys.path.append(r'c:\Users\rokas\Documents\Github\BCI\mi-bci\code')

import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from evaluation.within_session import train_and_evaluate, epochs_for_train
from helper_functions import setup_logger
from pathlib import Path

def cs_evaluate(epochs: dict,pipelines=None,n_splits=5,pipeline_name='single_pipeline',param_grid=None,
                  save_dir="results", save_model=False, log=None, single_pipeline_key=None, 
                  subject=None,n_jobs=1, feature_pipelines= None, 
                  classifier_pipelines = None, construct_pipelines=False,
                  train_run=1,test_run=2, result_join=True,return_within=False
                  ):
    
    if log is None:
        from helper_functions import setup_logger
        log = setup_logger('pipeline_evaluation')
    
    epochs_train = epochs[str(train_run)]
    epochs_eval = epochs[str(test_run)]
    
    if not save_model:
        save_dir = 'temp'
    os.makedirs(save_dir, exist_ok=True)
    
    if return_within:
        ws_2_results, model_2_filename_list = train_and_evaluate(epochs=epochs_train,
                                        pipelines=pipelines,
                                        n_splits=n_splits,
                                        pipeline_name=pipeline_name,
                                        param_grid=param_grid,
                                        save_dir=save_dir,
                                        save_model=False,
                                        log=log,
                                        single_pipeline_key=single_pipeline_key,
                                        subject=subject,
                                        run=test_run,
                                        n_jobs=n_jobs,
                                        feature_pipelines=feature_pipelines,
                                        classifier_pipelines=classifier_pipelines,
                                        construct_pipelines=construct_pipelines,
                                        return_model_filename = True
                                        )

    ws_results, model_filename_list = train_and_evaluate(epochs=epochs_train,
                                        pipelines=pipelines,
                                        n_splits=n_splits,
                                        pipeline_name=pipeline_name,
                                        param_grid=param_grid,
                                        save_dir=save_dir,
                                        save_model=True,
                                        log=log,
                                        single_pipeline_key=single_pipeline_key,
                                        subject=subject,
                                        run=train_run,
                                        n_jobs=n_jobs,
                                        feature_pipelines=feature_pipelines,
                                        classifier_pipelines=classifier_pipelines,
                                        construct_pipelines=construct_pipelines,
                                        return_model_filename = True
                                        )


    if log:
        log.info(f"---------------------------------------------------------------------------------")
        log.info(f">>>>>>>>>>>>>>>>>>>>>>BEGINNING TESTING MODEL TRANSFER<<<<<<<<<<<<<<<<<<<<<<<<<<")
        log.info(f"---------------------------------------------------------------------------------")

    if result_join:
        results = {
            'cross_session': {},
            'within_session' :{}
        }
    else:
        results = {
            'cross_session': {}
            }
    
    for rel_path in model_filename_list:
        full_path = os.path.join(os.getcwd(), rel_path)
        try:
            model = joblib.load(full_path)
            path = Path(rel_path)
            parts = path.stem.split('_')
            model_name = '_'.join(parts[:2])  
            if log:
                log.info(f"Loaded model from {rel_path}")
        except Exception as e:
            if log:
                log.error(f"Error loading model from {rel_path}: {e}")
            raise e

        # Evaluate the model on the provided test data
        try:
            epochs_train = epochs_for_train(epochs_eval,log)
            X = epochs_train.get_data(copy=False)
            y = epochs_train.events[:, -1] - 1
            
            # Generate predictions using the loaded model (using X not X_test, since data is in X, y)
            y_pred = model.predict(X)
            
            test_accuracy = accuracy_score(y, y_pred)
            correct_mask = (y_pred == y)  # boolean array, True if correct

            # Compute overall metrics
            test_accuracy = accuracy_score(y, y_pred)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                test_roc_auc = roc_auc_score(y, y_pred_proba)
            else:
                test_roc_auc = None

            # Build your eval_results dict
            eval_results = {
                'test_accuracy':    test_accuracy,
                'test_roc_auc':     test_roc_auc,
                'mean_test_accuracy': test_accuracy,
                'mean_test_auc':      test_roc_auc,

                # === per-epoch entries ===
                'correct_mask':     correct_mask,# numpy bool array
                'correct_indices':  np.where(correct_mask)[0].tolist(),
                'incorrect_indices': np.where(~correct_mask)[0].tolist(),
                'per_epoch': [
                    {'index': int(i),
                    'true_label': int(int_label),
                    'pred_label': int(pred_label),
                    'correct': bool(correct)}
                    for i, (int_label, pred_label, correct)
                    in enumerate(zip(y, y_pred, correct_mask))
                ]
            }

            if log:
                log.info(f"Overall accuracy: {test_accuracy:.4f}, AUC: {test_roc_auc}")
                log.info(f"Correct trials: {len(eval_results['correct_indices'])}, "
                        f"Incorrect: {len(eval_results['incorrect_indices'])}")  
        
            results['cross_session'][model_name] = eval_results

        except Exception as e:
            if log:
                log.error(f"Error evaluating the model on the provided test data {rel_path}: {e}")
            results['cross_session'][model_name] = None
            continue
        
    if result_join:
        results['within_session'][train_run] = ws_results
        if return_within:
            results['within_session'][test_run] = ws_2_results


    return results