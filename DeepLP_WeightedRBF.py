from __future__ import print_function
from DeepLP_RBF import DeepLP_RBF
import tensorflow as tf
import numpy as np

class DeepLP_WeightedRBF(DeepLP_RBF):

    def __init__(self, iter_, num_nodes, features, graph, sigma_, lr, session):
        phi          = tf.constant(features, dtype=tf.float32)
        G            = tf.constant(graph, dtype=tf.float32)
        self.sigma   = tf.constant(sigma_, dtype=tf.float32)
        num_features = features.shape[1]
        self.theta   = tf.Variable(tf.ones([1,num_features]))
        phi          = phi * self.theta
        self.W       = self.init_weights(phi, G)

        self.build_graph(iter_,lr,num_nodes,session)

    def save_params(self,epoch,data,n):
        thetab = self.eval(self.theta,data)[0]
        self.thetas.append(thetab)
        if epoch % 100 == 0:
            print("theta:",thetab)

    def train(self,data,epochs):
        self.thetas = []
        super().train(data,epochs)
