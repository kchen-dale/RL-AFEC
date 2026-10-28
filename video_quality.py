import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel('ERROR')
# Given 90-d non-reference video data, output video quality score 0-100

def build_model():
    model = keras.Sequential([
    layers.Dense(90, activation='relu', input_shape=[90]),
    layers.Dense(90, activation='relu'),
    layers.Dense(1)
  ])

    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weights-494--5.61865.hdf5') # choose the best checkpoint
_NN_model = None

def load_model():
    # Build the model once per process and reuse it; rebuilding it and reloading
    # the weights from disk on every call dominates the cost of a reward step.
    global _NN_model
    if _NN_model is None:
        _NN_model = build_model()
        _NN_model.load_weights(WEIGHTS_FILE)

    return _NN_model


