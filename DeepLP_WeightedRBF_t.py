from __future__ import print_function
from DeepLP_RBF import DeepLP_RBF
import tensorflow as tf
import numpy as np

class DeepLP_WeightedRBF_t(DeepLP_RBF):

    def __init__(self, iter_, num_nodes, features, graph, sigma_, theta_, lr, session):
        phi          = tf.constant(features, dtype=tf.float32)
        G            = tf.constant(graph, dtype=tf.float32)
        self.sigma   = tf.constant(sigma_, dtype=tf.float32)
        num_features = features.shape[1]
        self.theta   = tf.placeholder("float", shape=[1,4])
        phi          = phi * self.theta
        self.W       = self.init_weights(phi, G, sigma_)

        self.build_graph(iter_,lr,num_nodes,session)

    def save_params(self,epoch,data,n):
        thetab = self.eval(self.theta,data)
        self.thetas.append(thetab)
        if epoch % 10 == 0:
            print("theta:",thetab)

    def train(self,data,full_data,epochs):
        self.thetas = []
        super().train(data,full_data,epochs)

    def eval(self,tensor,data):
        return self.sess.run(tensor, feed_dict={self.X:data['X'],
                                    self.y:data['y'],
                                    self.unlabeled:data['unlabeled'],
                                    self.labeled:data['labeled'],
                                    self.masked:data['masked'],
                                    self.true_labeled:data['true_labeled'],
                                    self.theta:data['theta']
                                    })

    def build_graph(self,iter_,lr,num_nodes,session):
        self.sess   = session
        self.lr     = lr
        self.iter_  = iter_ # Layer size

        shape             = [None, num_nodes]
        self.X            = tf.placeholder("float", shape=shape)
        self.y            = tf.placeholder("float", shape=shape)
        self.unlabeled    = tf.placeholder("float", shape=shape)
        self.labeled      = tf.placeholder("float", shape=shape)
        self.masked       = tf.placeholder("float", shape=shape)
        self.true_labeled = tf.placeholder("float", shape=shape)

        self.yhat = self.forwardprop()
