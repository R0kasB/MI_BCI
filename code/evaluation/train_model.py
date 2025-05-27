import joblib
import os

def train_model(data, pipeline, log=None, model_name="trained_model", log_name="model_training", data_type=str):
    """
    Train a model using the provided pipeline and return the trained model.

    Parameters:
    - epochs: MNE Epochs object containing the data.
    - pipeline: The machine learning pipeline to train.
    - log: Logger instance for logging information (optional).
    - save_dir: Directory to save the trained model (optional).
    - save: Whether to save the trained model to a file (default: False).
    - model_name: Name of the saved model file (if save=True).

    Returns:
    - trained_model: The trained model.
    """
    if log is None:
        import logging
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(log_name)

    if data_type == 'epochs':
        log.info("Preparing data for training.")
        data_train = data.copy().crop(tmin=1.0, tmax=3.5)  # As in original article
        X = data_train.get_data(copy=False)
        y = data_train.events[:, -1] - 1

        log.info("Training the model.")
        trained_model = pipeline.fit(X, y)
        
    elif data_type == 'raw':
        log.info('code for raw type is yet to be added')

    else:
        log.error(f"Data type- {data_type} -is not supported")
    return trained_model

#epohom
    # log.info("Preparing data for training.")
    # data_train = data.copy().crop(tmin=1.0, tmax=3.5)  # As in original article
    # X = data_train.get_data(copy=False)
    # y = data_train.events[:, -1] - 1

    # log.info("Training the model.")
    # trained_model = pipeline.fit(X, y)
