import joblib
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
import numpy as np
import time
import psutil
import os
from sklearn.pipeline import Pipeline


def start_monitoring():
    """
    Record initial CPU time, wall-clock time, and memory usage.
    
    Returns:
        dict: A dictionary with keys 'cpu_start', 'wall_start', and 'mem_start'
    """
    process = psutil.Process(os.getpid())
    monitor_data = {
        'cpu_start': time.process_time(),      # CPU time in seconds
        'wall_start': time.perf_counter(),       # Wall-clock time in seconds
        'mem_start': process.memory_info().rss   # Memory usage in bytes
    }
    return monitor_data

def end_monitoring(monitor_data):
    """
    Compute differences in CPU time, wall-clock time, and memory usage.
    
    Args:
        monitor_data (dict): The dictionary returned by start_monitoring().
        
    Returns:
        str: A formatted string reporting the CPU time, wall-clock time, and memory change.
    """
    process = psutil.Process(os.getpid())
    cpu_end = time.process_time()
    wall_end = time.perf_counter()
    mem_end = process.memory_info().rss
    
    cpu_time_used = cpu_end - monitor_data['cpu_start']
    wall_time_used = wall_end - monitor_data['wall_start']
    mem_used = mem_end - monitor_data['mem_start']
    
    # Convert memory change from bytes to megabytes
    mem_used_mb = mem_used / (1024 * 1024)
    
    result = {"CPU_time_used(sec)": cpu_time_used,
              "Wall-clock_time_used(sec)": wall_time_used,
              "Memory_change(MB)": mem_used_mb}
    return result


def epochs_for_train(epochs,log):
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=3.5) #as in original article
    log.info(f"Original epochs shape: {epochs.get_data().shape}, using epochs_train shape: {epochs_train.get_data().shape}")
    return epochs_train

def log_scores(log, train_accs, train_aucs, val_accs, val_aucs, name):
    """Log per-fold and summary scores."""
    for fold_idx, (train_acc, train_auc, val_acc, val_auc) in enumerate(zip(train_accs, train_aucs, val_accs, val_aucs)):
        log.info(f"======= Fold {fold_idx} =======")
        log.info(f"Training Accuracy: {train_acc:.4f}, Training AUC: {train_auc:.4f}")
        log.info(f"Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}")
    log.info(f"{name} out-of-fold mean training accuracy: {np.mean(train_accs):.4f}")
    log.info(f"{name} out-of-fold mean training AUC: {np.mean(train_aucs):.4f}")
    log.info(f"{name} out-of-fold mean validation accuracy: {np.mean(val_accs):.4f}")
    log.info(f"{name} out-of-fold mean validation AUC: {np.mean(val_aucs):.4f}")

def train_and_evaluate(epochs, pipelines=None, n_splits=5, pipeline_name='single_pipeline',param_grid=None,
                       save_dir="results", save_model=False,save_best_model=False, log=None, single_pipeline_key=None, subject=None, run=None,
                       n_jobs=1, feature_pipelines= None, classifier_pipelines = None, construct_pipelines=False,
                       return_model_filename=False):
    print('=============================================================================')
    """single pipeline key should only be passed if 1 pipeline is being evaluated #reiktu sutvarkyt, kad ne single pipeline vardas butu o pipeline key"""
    if log is None:
        from helper_functions import setup_logger
        log = setup_logger('pipeline_evaluation')
    
    log.info("Begining training and evaluating pipelines")
    
    epochs_train = epochs_for_train(epochs,log)

    results = {}
    model_filename_list = []
#patikrint ar nekelia erroru copy=False, su raw meta
    X = epochs_train.get_data(copy=False)
   
    y = epochs_train.events[:, -1] - 1
    
    if pipelines or construct_pipelines:
        if not isinstance(pipelines, dict):
            pipelines = {pipeline_name: pipelines}

        for name, pipeline in pipelines.items():
            # kad nebūtų evaluating pipeline - single pipeline
            if single_pipeline_key:
                name = single_pipeline_key
            
            log.info(f"Evaluating pipeline: {name}")
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=529)

            # Define scoring metrics
            scoring = {
                'accuracy': 'accuracy',
                'roc_auc': 'roc_auc'
            }

            # If param_grid is provided, use GridSearchCV to tune hyperparameters
            if param_grid:
                log.info(">>> cv: %s", skf)
                log.info(">>> scoring: %r (type=%s)", scoring, type(scoring))
                log.info(">>> return_train_score: %s", True)
                log.info(">>> estimator type: %s", type(pipeline))

                # Use GridSearchCV if param_grid is provided
                pipeline = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=skf,
                    scoring=scoring,
                    refit='accuracy',  # Use 'accuracy' for refitting the best model
                    n_jobs=n_jobs,
                    return_train_score=True
                )
                pipeline.fit(X, y)

                best_model = pipeline.best_estimator_
                best_params = pipeline.best_params_
                cv_results = pipeline.cv_results_

                train_accs = cv_results['mean_train_accuracy']
                train_aucs = cv_results['mean_train_roc_auc']
                val_accs = cv_results['mean_test_accuracy']
                val_aucs = cv_results['mean_test_roc_auc']
                                 
            else:        
                results = {}
                
                # Perform cross-validation with training scores
                if construct_pipelines:
                    # Loop over each feature extraction pipeline.
                    for feat_name, feat_pipe in feature_pipelines.items():
                        # Compute features once for the current feature pipeline.
                        log.info(">>> Feature pipeline: %r, object: %r", feat_name, feat_pipe)
                        log.info("    params: %s", feat_pipe.get_params())
                        
                        # Loop over each classifier pipeline.
                        log.info(">>> Classifier pipelines:")
                        for clf_name, clf in classifier_pipelines.items():
                            
                            log.info("    %r -> %s, params=%s", clf_name, type(clf), clf.get_params())
                            # Construct a unique name for the combined pipeline.
                            pipeline_name = f"{feat_name}_{clf_name}"
                            full_pipe = Pipeline([
                                (f"{feat_name}", feat_pipe),
                                (f"{clf_name}",  clf)
                            ])
                            log.info("Evaluating pipeline: %s_%s", feat_name, clf_name)
                            try:
                                log.info(">>> cv: %s", skf)
                                log.info(">>> scoring: %r (type=%s)", scoring, type(scoring))
                                log.info(">>> return_train_score: %s", True)
                                log.info(">>> estimator type: %s", type(clf))
                                scores = cross_validate(
                                    estimator=full_pipe,
                                    X=X,
                                    y=y,
                                    cv=skf,
                                    scoring=scoring,
                                    return_train_score=True,
                                    n_jobs=n_jobs
                                )
                                
                                # resource_stats_class = end_monitoring(monitor_data_class)

                                # Extract test and train scores.
                                val_accs = scores['test_accuracy']
                                val_aucs = scores['test_roc_auc']
                                train_accs = scores['train_accuracy']
                                train_aucs = scores['train_roc_auc']
                                
                                # Log the scores using the combined pipeline name.
                                log_scores(log, train_accs, train_aucs, val_accs, val_aucs, pipeline_name)
                                
                                # resource_stats_pipeline = {
                                #     "CPU_time_used(sec)": resource_stats_feat["CPU_time_used(sec)"] + resource_stats_class["CPU_time_used(sec)"],
                                #     "Wall-clock_time_used(sec)": resource_stats_feat["Wall-clock_time_used(sec)"] + resource_stats_class["Wall-clock_time_used(sec)"],
                                #     "Memory_change(MB)": resource_stats_feat["Memory_change(MB)"] + resource_stats_class["Memory_change(MB)"]
                                # }
                                
                                # Save results under the constructed name.
                                results[pipeline_name] = {
                                    'train_accuracy': train_accs,
                                    'train_roc_auc': train_aucs,
                                    'val_accuracy': val_accs,
                                    'val_roc_auc': val_aucs,
                                    'mean_train_accuracy': np.mean(train_accs),
                                    'mean_train_auc': np.mean(train_aucs),
                                    'mean_val_accuracy': np.mean(val_accs),
                                    'mean_val_auc': np.mean(val_aucs)
                                    # 'feat_resources_used': resource_stats_feat,
                                    # 'class_resources_used': resource_stats_class,
                                    # 'recourses_per_pipeline': resource_stats_pipeline
                                }
                            except Exception as e:
                                log.error(f"An error occurred in pipeline (2nd try) {pipeline_name}: {e}")
                                results[pipeline_name] = None # taip geriau tikriausiai
                                continue
                            if save_model:
                                try:
                                    full_pipe.fit(X, y)
                                    log.info(f'saving {pipeline_name} model')
                                    os.makedirs(save_dir, exist_ok=True)
                                    model_filename = os.path.join(save_dir, f'{pipeline_name}_{name}_{subject}-{run}.pkl')
                                    joblib.dump(full_pipe, model_filename)
                                    log.info(f"{pipeline_name} model saved as {model_filename}")
                                    model_filename_list.append(model_filename)
                                except Exception as e:
                                    log.error(f"An error occurred while saving {pipeline_name}: {e}")


                else:
                    scores = cross_validate(
                        estimator=pipeline,
                        X=X,
                        y=y,
                        cv=skf,
                        scoring=scoring,
                        return_train_score=True,  # Include training scores
                        n_jobs=n_jobs
                    )
                # Extract test and train scores
                val_accs = scores['test_accuracy']
                val_aucs = scores['test_roc_auc']
                train_accs = scores['train_accuracy']
                train_aucs = scores['train_roc_auc']
                
                best_model = pipeline
                best_params = None
                cv_results = None
                
            log_scores(log, train_accs, train_aucs, val_accs, val_aucs, name)

            # Save the best model
            if save_best_model:
                os.makedirs(save_dir, exist_ok=True)
                model_filename = os.path.join(save_dir, f'best_model_{name}_{subject}-{run}.pkl')
                joblib.dump(best_model, model_filename)
                log.info(f"Best model saved as {model_filename}")

                # Save the best hyperparameters (if available)
                if best_params:
                    params_filename = os.path.join(save_dir, f'best_params_{name}.json')
                    with open(params_filename, 'w') as f:
                        json.dump(best_params, f)
                    log.info(f"Best hyperparameters saved as {params_filename}")

            results[name] = {
                'train_accuracy': train_accs,
                'train_roc_auc': train_aucs,
                'val_accuracy': val_accs,
                'val_roc_auc': val_aucs,
                'mean_train_accuracy': np.mean(train_accs),
                'mean_train_auc': np.mean(train_aucs),
                'mean_val_accuracy': np.mean(val_accs),
                'mean_val_auc': np.mean(val_aucs),
                # 'best_model': best_model,
                'best_params': best_params
                #pridėt best model scores
            }
            if param_grid:
                results[name]['cv_results'] = cv_results
            # else:
            #     results[name]['scores'] = scores #galimai erorai kyla poto results bandant atsidaryt

                
        except Exception as e:
            log.error(f"An error occurred in pipeline (1st try) {name}: {e}")
            results[name] = None # taip geriau tikriausiai
            #results[name] = {'error': str(e)}
# sukurt temporary saving, jei užlužtu kažkas ir nebaigtų visų pipelinų vertint.
    if return_model_filename:
        return results, model_filename_list
    return results


'''
#manual
def train_and_evaluate(train, pipelines, n_splits=5):
    results = {}

    for name, pipeline in pipelines.items():
        print(f"Evaluating pipeline: {name}")
        try:
            sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=529)

            X, y, groups = get_X_y(train)

            fold = 0
            aucs = []
            accs = []
            for train_idx, val_idx in sgk.split(X, y, groups):
                X_tr = X.loc[train_idx]
                y_tr = y.loc[train_idx]
                
                X_val = X.loc[val_idx]
                y_val = y.loc[val_idx]

                # Fit Model on Train
                clf = pipeline
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_val)
                pred_prob = clf.predict_proba(X_val)[:, 1]
                acc_score = accuracy_score(y_val, pred)
                auc_score = roc_auc_score(y_val, pred_prob)

                print(f"======= Fold {fold} ========")
                print(
                    f"Our accuracy on the validation set is {acc_score:0.4f} and AUC is {auc_score:0.4f}"
                )
                fold += 1
                aucs.append(auc_score)
                accs.append(acc_score)
            auc = np.array(aucs)
            acc = np.array(accs)
            print(f'Our out of fold AUC score is {auc:0.4f}')
            print(f'Our out of fold ACC score is {acc:0.4f}')

            print(f"Scores: {acc}")
            print(f"Mean accuracy: {np.mean(acc):.4f}")
            results[name] = auc, acc 
        except Exception as e:
            print(f"An error occurred in pipeline {name}: {e}")
            results[name] = None
    return results
    
'''