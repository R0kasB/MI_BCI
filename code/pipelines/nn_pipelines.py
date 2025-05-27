# TensorFlow/Keras Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Scikit-learn Imports
from sklearn.pipeline import Pipeline

# Custom Transformer
from pipelines.nn_transformers import EEGNetTransformer

# Other Libraries
import numpy as np  # Ensure NumPy is imported for array manipulations


# ===========================================
# Neural Network Pipelines
# ===========================================

nn_pipelines = {}

def create_eegnet_model(n_channels, n_times, n_classes):
    """Creates an EEGNet model."""
    input_shape = (n_channels, n_times, 1)
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(16, (1, 64), padding='same',
               use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((n_channels, 1), use_bias=False,
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = SeparableConv2D(32, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Placeholder function to create the model with the correct input dimensions
def create_eegnet_model_wrapper():
    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))
    return create_eegnet_model(n_channels, n_times, n_classes)


# 1. EEGNet (Deep CNN)
nn_pipelines['EEGNet_CNN'] = Pipeline([
    ('reshape', EEGNetTransformer()),
    ('eegnet', KerasClassifier(build_fn=create_eegnet_model_wrapper, epochs=50, batch_size=16, verbose=0))
])