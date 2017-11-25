from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP_RBF import DeepLP_RBF

class DeepLP_WeightedRBF(DeepLP_RBF):

    def __init__(self, iter_, num_nodes, features, graph, sigma_, theta_, lr, regularize=0):
        phi          = tf.constant(features, dtype=tf.float32)
        G            = tf.constant(graph, dtype=tf.float32)
        self.sigma   = tf.constant(sigma_, dtype=tf.float32)
        num_features = features.shape[1]
        self.theta   = tf.Variable(tf.convert_to_tensor(theta_, dtype=tf.float32))
        phi          = phi * self.theta
        self.W       = self.init_weights(phi, G, sigma_)
        self.regularize = regularize

        self.build_graph(iter_,lr,num_nodes)

    def save_params(self,epoch,data,n):
        thetab = self.get_val(self.theta)
        self.thetas.append(thetab)
        if epoch % 10 == 0:
            print("theta:",thetab)

    def train(self,data,full_data,epochs):
        self.thetas = []
        super().train(data,full_data,epochs)

    def plot_params(self):
        plt.plot(self.thetas)
        plt.title("parameters")
        plt.show()
