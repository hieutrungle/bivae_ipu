import tensorflow as tf
import numpy as np
import os
import errno
from utils.normalizer import DataNormalizer


class Data():
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_added_data = 0
        self.data_dim = None
        self.steps_per_execution = 0
        self.normalizer = None

    def load_data(self, data_name):
        if data_name == "mnist":
            data = self.load_mnist()
        elif data_name == "cesm":
            data = self.load_cesm()
        elif data_name == "isabel":
            data = self.load_isabel()
        else:
            raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), data_name)

        # Make data_len divisible by batch_size

        data = self.complete_data_with_batch(data)
        self.data_dim = data.shape

        data = self.to_tf_dataset(data)
        return data.prefetch(tf.data.AUTOTUNE)

    def load_mnist(self):

        self.normalizer = DataNormalizer(min_scale=0.0, max_scale=1.0)

        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        data = np.vstack([x_train, x_test])
        data = self.normalizer.normalize_minmax(data)
        data = data[..., np.newaxis]

        return data

    def complete_data_with_batch(self, data):

        len_last_batch = data.shape[0] - data.shape[0] // self.batch_size * self.batch_size
        num_added_data = self.batch_size - len_last_batch
        self.num_added_data = num_added_data
        number_of_rows = data.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_added_data, replace=False)
        additional_imgs = data[random_indices]
        data = np.append(data, additional_imgs, axis=0)
        return data

    def to_tf_dataset(self, data, dtype=tf.float32):
        data = tf.convert_to_tensor(data, dtype=dtype)
        data = tf.data.Dataset.from_tensor_slices((data, data))
        data = data.batch(self.batch_size, drop_remainder=True)
        return data
