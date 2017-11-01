from __future__ import print_function
import tensorflow as tf
import numpy as np

class DeepLP:
    def __init__(self, iter_, num_nodes, weights, lr, session):
        self.W      = self.init_weights(weights)
        self.build_graph(iter_,lr,num_nodes,session)

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
        self.backwardprop()

    def init_weights(self,weights_np):
        """ Weight initialization """
        weights = tf.convert_to_tensor(weights_np, np.float32)
        return tf.Variable(weights)

    def get_value(self,val):
        return self.sess.run(val)

    def forwardprop(self):
        T = self.W / tf.reduce_sum(self.W, axis = 0)
        Tnorm = tf.transpose(T / tf.reduce_sum(T, axis = 1))

        trueX = self.X
        X = self.X

        for i in range(self.iter_):
            h = tf.tensordot(X, Tnorm, axes = 1)
            h = tf.multiply(h, self.unlabeled) + tf.multiply(trueX, self.labeled)
            X = h
        return h

    def backwardprop(self):
        # Backward propagation
        self.loss = self.calc_loss(self.masked,self.y,self.yhat)
        self.updates = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def calc_loss(self,mask,y,yhat):
        loss_mat = tf.multiply(mask, (y-yhat) ** 2 )
        return tf.reduce_sum(loss_mat) / tf.count_nonzero(loss_mat,dtype=tf.float32)

    def calc_accuracy(self,y,yhat):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(yhat),y),tf.float32))


    def labelprop(self,data):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        pred = self.eval(self.yhat,data)
        return pred

    def eval(self,tensor,data):
        return self.sess.run(tensor, feed_dict={self.X:data['X'],
                                    self.y:data['y'],
                                    self.unlabeled:data['unlabeled'],
                                    self.labeled:data['labeled'],
                                    self.masked:data['masked'],
                                    self.true_labeled:data['true_labeled']})

    def save_params(self,epoch,data,n):
        pass

    def save(self,epoch,data,n):
        labeled_loss = self.eval(self.loss,data)
        unlabeled_loss = self.eval(self.calc_loss((1-self.true_labeled),self.y,self.yhat),data)
        accuracy = self.eval(self.calc_accuracy(self.y,self.yhat),data)
        self.labeled_losses.append(labeled_loss)
        self.unlabeled_losses.append(unlabeled_loss)
        self.accuracies.append(accuracy)
        if epoch % 1 == 0 or epoch == -1:
            print("epoch:",epoch,"labeled loss:",labeled_loss,"unlabeled loss:",unlabeled_loss,"accuracy:",accuracy)
        self.save_params(epoch,data,n)

    def train(self,data,epochs):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        n = len(data['X'])
        self.labeled_losses = []
        self.unlabeled_losses = []
        self.accuracies = []
        self.save(-1,data,n)

        for epoch in range(epochs):
            # Train with each example
            for i in range(n):
                self.eval(self.updates,data)
            self.save(epoch,data,n)
