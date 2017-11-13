from __future__ import print_function
import tensorflow as tf
import numpy as np

class DeepLP_Sparse:
    def __init__(self, iter_, num_nodes, weights, lr, session, regularize=0):
        self.W      = self.init_weights(weights)
        self.regularize = regularize
        self.build_graph(iter_,lr,num_nodes,session)


    def build_graph(self,iter_,lr,num_nodes,session):
        self.sess   = session
        self.lr     = lr
        self.iter_  = iter_ # Layer size

        shape             = [None, num_nodes]
        self.X            = tf.sparse_placeholder("float", shape=shape)
        self.y            = tf.sparse_placeholder("float", shape=shape)
        self.unlabeled    = tf.sparse_placeholder("float", shape=shape)
        self.labeled      = tf.sparse_placeholder("float", shape=shape)
        self.masked       = tf.sparse_placeholder("float", shape=shape)
        self.true_labeled = tf.sparse_placeholder("float", shape=shape)

        self.yhat = self.forwardprop()
        self.backwardprop()

    def init_weights(self,weights_np):
        """ Weight initialization """
        weights = self.convert_to_sparse_tensor(weights_np)
        return tf.Variable(weights)

    def convert_to_sparse_tensor(self,np_mat):
        weights = tf.convert_to_tensor(np_mat, np.float32)
        idx = tf.where(tf.not_equal(weights, 0))
        sparse = tf.SparseTensor(idx, tf.gather_nd(weights, idx), weights.get_shape())
        return sparse

    def get_value(self,val):
        return self.sess.run(val)

    def sparse_transpose(self,mat):
        input_st = mat
        indices = input_st.indices
        # Reverse the columns of `indices`
        transposed_indices = tf.concat(1, [indices[:, 1:2], indices[:, 0:1]])

        dense_shape = input_st.dense_shape
        # Reverse the elements of `dense_shape`
        transposed_dense_shape = tf.pack([dense_shape[1], dense_shape[0]])

        transposed_st = tf.SparseTensor(transposed_indices, input_st.values, transposed_dense_shape)
        return transposed_st

    def forwardprop(self):
        T = self.W / tf.sparse_reduce_sum(self.W, axis = 0)

        Tnorm = self.sparse_transpose(T / tf.sparse_reduce_sum(T, axis = 1))

        trueX = self.X
        X = self.X

        for i in range(self.iter_):
            h = tf.sparse_matmul(X, Tnorm)
            h = h.__mul__(self.unlabeled) + trueX.__mul__(self.labeled)
            X = h
        return h

    def backwardprop(self):
        # Backward propagation
        if self.regularize:
            self.loss = self.calc_loss(self.masked,self.y,self.yhat) + self.regularize*tf.nn.l2_loss(self.theta-1)
        else:
            self.loss = self.calc_loss(self.masked,self.y,self.yhat)
        self.updates = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def calc_loss(self,mask,y,yhat):
        loss_mat = tf.multiply(mask, (y-yhat) ** 2 )
        return tf.reduce_sum(loss_mat) / tf.count_nonzero(loss_mat,dtype=tf.float32)

    def calc_accuracy(self,y,yhat,full=False):
        if full:
            acc_mat = tf.multiply((1-self.true_labeled),tf.cast(tf.equal(tf.round(yhat),y),tf.float32))
            return tf.reduce_sum(acc_mat) / tf.count_nonzero((1-self.true_labeled),dtype=tf.float32)
        else:
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

    def save(self,epoch,data,full_data,n):
        labeled_loss = self.eval(self.loss,data)
        unlabeled_loss = self.eval(self.calc_loss((1-self.true_labeled),self.y,self.yhat),data)
        accuracy = self.eval(self.calc_accuracy(self.y,self.yhat),data)
        sol_unlabeled_loss = self.eval(self.calc_loss((1-self.true_labeled),self.y,self.yhat),full_data)
        sol_accuracy = self.eval(self.calc_accuracy(self.y,self.yhat,True),full_data)
        acc_mat = tf.multiply((1-self.true_labeled),tf.cast(tf.equal(tf.round(self.yhat),self.y),tf.float32))
        self.labeled_losses.append(labeled_loss)
        self.unlabeled_losses.append(unlabeled_loss)
        self.accuracies.append(accuracy)
        self.sol_accuracies.append(sol_accuracy)
        self.sol_unlabeled_losses.append(sol_unlabeled_loss)
        if epoch % 10 == 0 or epoch == -1:
            print("epoch:",epoch,"labeled loss:",labeled_loss,"unlabeled loss:",unlabeled_loss,"accuracy:",accuracy,"sol unlabeled loss:",sol_unlabeled_loss,"sol accuracy:",sol_accuracy)
        self.save_params(epoch,data,n)

    def train(self,data,full_data,epochs):
        init = tf.global_variables_initializer()
        self.sess.run(init)
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
                self.eval(self.updates,data)
            self.save(epoch,data,full_data,n)
