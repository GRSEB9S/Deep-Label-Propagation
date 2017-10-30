from __future__ import print_function
from DeepLP import DeepLP
import tensorflow as tf
import numpy as np


class DeepLP_RBF(DeepLP):

    def __init__(self, iter_, num_nodes, features, graph, sigma, lr, session):
        phi         = tf.constant(features, dtype=tf.float32)
        G           = tf.constant(graph, dtype=tf.float32)
        self.W      = self.init_weights(phi, G, sigma)

        super().build_graph(iter_,lr,num_nodes,session)

    def print_train(self,epoch,data):
        lossb =　super().eval(self.loss,data)
        sigmab = super().eval(self.sigma,data)

        print("epoch:",epoch,"loss:",lossb,"sigma:",sigmab)

    def init_weights(self, phi, G, sigma):
        self.sigma = tf.Variable(sigma, dtype=tf.float32)
        r = tf.reduce_sum(phi*phi, 1)
        r = tf.reshape(r, [-1, 1])
        D = tf.cast(r - 2*tf.matmul(phi, tf.transpose(phi)) + tf.transpose(r),tf.float32)
        W = tf.exp(-tf.divide(D, self.sigma ** 2)) * G
        return W
