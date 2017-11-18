from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class DeepLP:
    def __init__(self, iter_, num_nodes, weights, lr, regularize=0):
        self.W      = self.init_weights(weights)
        self.regularize = regularize
        self.build_graph(iter_,lr,num_nodes)


    def build_graph(self,iter_,lr,num_nodes):
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

    def get_val(self,val):
        return self.sess.run(val)

    def forwardprop(self):
        T = self.W / tf.reduce_sum(self.W, axis = 0, keep_dims=True)
        Tnorm = T / tf.reduce_sum(T, axis = 1, keep_dims=True)

        trueX = self.X
        X = self.X

        def layer(i,X,trueX,Tnorm):
            h = X @ Tnorm
            h = tf.multiply(h, self.unlabeled) + tf.multiply(trueX, self.labeled)
            return [i+1,h,trueX,Tnorm]

        def condition(i,X,trueX,Tnorm):
            return self.iter_ > i

        _,h,_,_ = tf.while_loop(condition, layer, loop_vars=[0,X,trueX,Tnorm])
        return h

    def backwardprop(self):
        # Backward propagation
        if self.regularize:
            self.loss = self.calc_loss(self.masked,self.y,self.yhat) + self.regularize*tf.nn.l2_loss(self.theta-1)
        else:
            self.loss = self.calc_loss(self.masked,self.y,self.yhat)
        self.updates = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.unlabeled_loss     = self.calc_loss((1-self.true_labeled),self.y,self.yhat)
        self.accuracy           = self.calc_accuracy(self.y,self.yhat)
        self.sol_unlabeled_loss = self.calc_loss((1-self.true_labeled),self.y,self.yhat)
        self.sol_accuracy       = self.calc_accuracy(self.y,self.yhat,True)

    def calc_loss(self,mask,y,yhat):
        loss_mat = tf.multiply(mask, (y-yhat) ** 2 )
        return tf.reduce_sum(loss_mat) / tf.count_nonzero(loss_mat,dtype=tf.float32)

    def calc_accuracy(self,y,yhat,full=False):
        if full:
            acc_mat = tf.multiply((1-self.true_labeled),tf.cast(tf.equal(tf.round(yhat),y),tf.float32))
            return tf.reduce_sum(acc_mat) / tf.count_nonzero((1-self.true_labeled),dtype=tf.float32)
        else:
            return tf.reduce_mean(tf.cast(tf.equal(tf.round(yhat),y),tf.float32))

    def open_sess(self):
        self.sess   = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def close_sess(self):
        self.sess.close()

    def labelprop(self,data):
        self.open_sess()
        pred = self.eval(self.yhat,data)
        # self.close_sess()

        return pred

    def eval(self,vals,data):
        return self.sess.run(vals, feed_dict={self.X:data['X'],
                                    self.y:data['y'],
                                    self.unlabeled:data['unlabeled'],
                                    self.labeled:data['labeled'],
                                    self.masked:data['masked'],
                                    self.true_labeled:data['true_labeled']})

    def save_params(self,epoch,data,n):
        pass

    def save(self,epoch,data,full_data,n):

        labeled_loss,unlabeled_loss,accuracy = self.eval([self.loss,self.unlabeled_loss,self.accuracy],data)
        sol_accuracy,sol_unlabeled_loss      = self.eval([self.sol_accuracy,self.sol_unlabeled_loss],full_data)
        self.labeled_losses.append(labeled_loss)
        self.unlabeled_losses.append(unlabeled_loss)
        self.accuracies.append(accuracy)
        self.sol_accuracies.append(sol_accuracy)
        self.sol_unlabeled_losses.append(sol_unlabeled_loss)
        if epoch % 10 == 0 or epoch == -1:
            print("epoch:",epoch,"labeled loss:",labeled_loss,"unlabeled loss:",unlabeled_loss,"accuracy:",accuracy,"sol unlabeled loss:",sol_unlabeled_loss,"sol accuracy:",sol_accuracy)
        self.save_params(epoch,data,n)

    def train(self,data,full_data,epochs):
        self.open_sess()

        n = len(data['X'])
        self.labeled_losses = []
        self.unlabeled_losses = []
        self.accuracies = []
        self.sol_accuracies = []
        self.sol_unlabeled_losses = []
        self.save(-1,data,full_data,n)
        for epoch in range(epochs):
            # Train with each example
            for i in range(n):
                start = time.time()
                self.eval(self.updates,data)
            self.save(epoch,data,full_data,n)
        # self.close_sess()

    def plot_loss(self,):
        plt.plot(self.labeled_losses,label="labeled loss")
        plt.plot(self.unlabeled_losses,label="unlabeled loss")
        plt.plot(self.sol_unlabeled_losses,label='validation unlabeled loss')
        plt.title("loss")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.accuracies,label="DeepLP, train")
        plt.plot([self.sol_accuracies[0]] * len(self.accuracies),label="LP")
        plt.plot(self.sol_accuracies,label="DeepLP, validation")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def plot_params(self):
        pass

    def plot(self):
        self.plot_loss()
        self.plot_accuracy()
        self.plot_params()
