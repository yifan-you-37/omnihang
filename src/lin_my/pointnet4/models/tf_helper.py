import tensorflow as tf
import numpy as np

def normalize_tensor(x):
    x_max = tf.reduce_max(x, axis=-1, keepdims=True)
    x_min = tf.reduce_min(x, axis=-1, keepdims=True)
    x_normalized = (x - x_min) / (x_max - x_min)

    return x_normalized