from __future__ import print_function
from DeepLP import DeepLP
import tensorflow as tf
import numpy as np


class DeepLP_RBF(DeepLP):

    def __init__(self, iter_, num_nodes, features, graph, sigma, lr, regularize=0):
        self.phi         = tf.constant(features, dtype=tf.float32)
        self.G           = tf.constant(graph, dtype=tf.float32)
        self.sigma  = tf.Variable(sigma, dtype=tf.float32)
        self.W           = self.init_weights(self.phi, self.G, self.sigma)
        self.regularize  = regularize

        self.build_graph(iter_,lr,num_nodes)

    def save_params(self,epoch,data,n):
        sigmab = self.get_val(self.sigma)
        self.sigmas.append(sigmab)
        if epoch % 100 == 0:
            print("sigma:",sigmab)

    def init_weights(self, phi, G, sigma):
        r = tf.reduce_sum(phi*phi, 1)
        r = tf.reshape(r, [-1, 1])
        D = tf.cast(r - 2*tf.matmul(phi, tf.transpose(phi)) + tf.transpose(r),tf.float32)
        W = tf.exp(-tf.divide(D, sigma ** 2)) * G
        return W

    def train(self,data,full_data,epochs):
        self.sigmas = []
        super().train(data,full_data,epochs)

    def pred(self,data,sigma):
        self.W = self.init_weights(self.phi,self.G,sigma)
        pred = self.eval(self.yhat,data)
        return pred
