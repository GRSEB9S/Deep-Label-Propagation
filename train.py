from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from LP import LP
from DeepLP import DeepLP
from DeepLP_RBF import DeepLP_RBF
from DeepLP_WeightedRBF import DeepLP_WeightedRBF
from DeepLP_WeightedRBF_t import DeepLP_WeightedRBF_t

from utils import *

# get labels, features and weights
LX, Ly, UX, Uy, Uy_sol = get_iris_data()
features = np.vstack((LX,UX))
true_labels = np.hstack((Ly,Uy_sol))
weights = rbf_kernel(features)
num_features = features.shape[1]
num_labeled = len(Ly)
num_unlabeled = len(Uy)
num_nodes = num_labeled+num_unlabeled
graph = (weights > 0).astype(int)

# prepare features for NN
LY = np.tile(Ly,(Ly.shape[0],1))
np.fill_diagonal(LY, 0.5)
UY = np.tile(Uy,(Ly.shape[0],1))

masked_ = np.hstack((np.identity(LY.shape[0]),np.zeros((Ly.shape[0],Uy.shape[0]))))
true_labeled = np.array([1] * LY.shape[0] + [0] * Uy.shape[0]).reshape(1,100)

unlabeled_test = np.hstack((np.zeros(LY.shape[0]),np.ones((Uy.shape[0])))).reshape(1,100)
test_data = {
    'X': np.hstack((Ly,Uy)).reshape(1,100),
    'y': np.tile(true_labels,(Ly.shape[0],1))[1:2],
    'unlabeled': unlabeled_test,
    'labeled': (1 - unlabeled_test).reshape(1,100),
    'true_labeled': true_labeled,
    'masked':masked_[0:1]
}

unlabeled_ = np.hstack((np.identity(LY.shape[0]),np.ones((Ly.shape[0],Uy.shape[0]))))
data = {
    'X':np.hstack((LY, UY)),
    'y':np.reshape(true_labels,(1,len(true_labels))),
    'unlabeled':unlabeled_,
    'labeled':1-unlabeled_,
    'true_labeled': true_labeled,
    'masked':masked_
}

import time
start = time.time()
import csv
import sys

w = csv.writer(open('no_reg.csv', 'a'), delimiter=',')
w1 = csv.writer(open('reg.csv', 'a'), delimiter=',')

theta = np.random.uniform(0,4.5,4)
print("----------------------------------")
print("theta:",str(i),theta)
print("----------------------------------")
sess = tf.Session()

dlp_wrbf = DeepLP_WeightedRBF(10, num_nodes, features, graph, np.var(features), theta, 0.01, sess, 0)
dlp_wrbf.train(data,test_data,100)

w.writerow([dlp_wrbf.accuracies[-1],dlp_wrbf.labeled_losses[-1],dlp_wrbf.unlabeled_losses[-1],dlp_wrbf.sol_accuracies[-1],dlp_wrbf.sol_unlabeled_losses[-1]])

end = time.time()
print(end - start)
print("----------------------------------")
dlp_wrbf.sess.close()
sess.close()

sess = tf.Session()
dlp_wrbf = DeepLP_WeightedRBF(10, num_nodes, features, graph, np.var(features), theta, 0.01, sess, 0.1)
dlp_wrbf.train(data,test_data,100)

w1.writerow([dlp_wrbf.accuracies[-1],dlp_wrbf.labeled_losses[-1],dlp_wrbf.unlabeled_losses[-1],dlp_wrbf.sol_accuracies[-1],dlp_wrbf.sol_unlabeled_losses[-1]])


end = time.time()
print(end - start)

dlp_wrbf.sess.close()
sess.close()
