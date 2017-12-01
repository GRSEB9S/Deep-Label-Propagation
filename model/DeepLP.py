from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import time

class DeepLP:
    '''
    Deep label propagation for predicting labels for unlabeled nodes.
    See our paper for details.
    '''
    def __init__(self, num_iter,
                       num_nodes,
                       weights,
                       lr,
                       regularize=0,       # add L1 regularization to loss
                       graph_sparse=False, # make the graph sparse
                       print_freq=10,      # print frequency when training
                       multi_class=False): # implementation for multiclass
        self.weights      = self._init_weights(weights)

        self._build_graph(self.weights,
                          num_iter,
                          num_nodes,
                          weights,
                          lr,
                          regularize,
                          graph_sparse,
                          print_freq,
                          multi_class)

    def _build_graph(self,num_iter,
                         num_nodes,
                         weights,
                         lr,
                         regularize,
                         graph_sparse,
                         print_freq,
                         multi_class):
        self.start_time   = time.time()

        # set instance variables
        self.num_iter     = num_iter
        self.lr           = lr
        self.regularize   = regularize
        self.graph_sparse = graph_sparse
        self.print_freq   = print_freq
        self.multi_class  = multi_class

        # initialize placeholders
        shape                       = [None, num_nodes]
        self.X                      = tf.placeholder("float", shape=shape)
        self.y                      = tf.placeholder("float", shape=shape)
        self.labeled_indices        = tf.placeholder("float", shape=shape)
        self.unlabeled_indices      = tf.placeholder("float", shape=shape)
        self.true_labeled_indices   = tf.placeholder("float", shape=shape)
        self.true_unlabeled_indices = tf.placeholder("float", shape=shape)

        self.yhat                   = self._forwardprop(self.X,
                                                        self.weights,
                                                        self.labeled_indices,
                                                        self.num_iter)
        self.update, self.metrics   = self._backwardprop(self.y,
                                                         self.yhat,
                                                         self.labeled_indices,
                                                         self.unlabeled_indices,
                                                         self.regularize,
                                                         self.lr)

    def _get_val(self,val):
        return self.sess.run(val)

    def _init_weights(self,weights_np):
        """ Weight initialization. """
        weights = tf.convert_to_tensor(weights_np, np.float32)
        return tf.Variable(weights)

    def _tnorm(weights):
        T = weights / tf.reduce_sum(weights, axis = 0, keep_dims=True)
        Tnorm = T / tf.reduce_sum(T, axis = 1, keep_dims=True)
        return Tnorm

    def _forwardprop(self, X,
                           weights,
                           labeled_indices,
                           num_iter):
        '''
        Forward prop which mimicks LP.
        '''

        Tnorm = self._tnorm(weights)

        def layer(i,h,X,Tnorm):
            # propagate labels
            h = tf.matmul(Tnorm,h)
            # don't update labeled nodes
            h[labeled_indices] = X[labeled_indices]
            return [i+1,h,X,Tnorm]

        def condition(i,X,trueX,Tnorm):
            return num_iter > i

        _,yhat,_,_ = tf.while_loop(condition, layer, loop_vars=[0,X,X,Tnorm])
        return yhat

    def _regularize_loss(self):
        return 0
        # tf.nn.l2_loss(self.theta-1)

    def _backwardprop(self, y,
                            yhat,
                            labeled_indices,
                            unlabeled_indices,
                            regularize,
                            lr):
        '''
        Backprop on unlabeled + masked labeled nodes.
        Calculate loss and accuracy for both train and validation dataset.
        '''
        # backward propagation
        loss          = (self._calc_loss(y,yhat,unlabeled_indices)
                            + regularize * self._regularize_loss())
        updates       = tf.train.AdamOptimizer(lr).minimize(loss)

        # evaluate performance
        accuracy      = self._calc_accuracy(y,yhat,unlabeled_indices)
        true_loss     = self._calc_loss(y,yhat,labeled_indices)
        true_accuracy = self._calc_accuracy(y,yhat,labeled_indices)

        metrics = [loss, accuracy, true_loss, true_accuracy]

        return updates, metrics

    def _calc_loss(self,y,yhat,indices):
        loss_mat = (y[indices]-yhat[indices]) ** 2
        return tf.reduce_mean(loss_mat)

    def _calc_accuracy(self,y,prob,indices):
        if self.multi_class:
            print(y,prob)
            yhat = tf.to_float(tf.equal(prob,tf.reduce_max(prob,axis=1)))
            return tf.reduce_all(tf.equal(yhat,y),axis=1)
        else:
            acc_mat = tf.cast(tf.equal(tf.round(yhat[indices]),y[indices]),tf.float32)
            return tf.reduce_sum(acc_mat) / tf.count_nonzero((1-self.true_labeled),dtype=tf.float32)

    def _open_sess(self):
        self.sess = tf.Session()
        init      = tf.global_variables_initializer()
        self.sess.run(init)

    def labelprop(self,data):
        self._open_sess()
        pred = self._eval(self.yhat,data)
        return pred

    def _eval(self,vals,data):
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
        if epoch % 1 == 0 or epoch == -1:
            print("epoch:",epoch,"labeled loss:",labeled_loss,"unlabeled loss:",unlabeled_loss,"accuracy:",accuracy,"sol unlabeled loss:",sol_unlabeled_loss,"sol accuracy:",sol_accuracy)
            print("--- %s seconds ---" % (time.time() - self.start_time))
            self.start_time = time.time()
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
            self.eval(self.updates,data)
            # print("--- %s seconds ---" % (time.time() - self.start_time))
            # self.start_time = time.time()
            self.save(epoch,data,full_data,n)
        # self.close_sess()

    def plot_loss(self):
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
