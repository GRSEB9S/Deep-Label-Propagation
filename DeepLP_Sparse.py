from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class DeepLP_Sparse:

    def dense_to_sparse(self,a_t):
        idx = tf.where(tf.not_equal(a_t, 0))
        # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
        sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
        return sparse

    def init_weights(self,weights_np):
        """ Weight initialization """
        weights = tf.convert_to_tensor(weights_np, np.float32)
        sparse_weights = self.dense_to_sparse(weights)
        return tf.Variable(sparse_weights)
