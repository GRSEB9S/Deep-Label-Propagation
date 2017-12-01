import sklearn
import pandas as pd
import numpy as np

class Data:
    '''
    Load datasets.
    '''
    def load_iris():
        # load iris data
        iris   = sklearn.datasets.load_iris()
        data   = iris["data"]
        labels = iris["target"]

        # get label 0 and 1, and corresponding features
        true_labels = labels[labels < 2]
        features = data[np.where(labels < 2)]

        return true_labels, features

    def load_cora():
        # load cora data
        nodes = pd.read_csv('cora/selected_contents.csv',index_col=0,)
        graph = np.loadtxt('cora/graph.csv',delimiter=',')
        id_    = np.array(nodes.index)

        # get label 0 and 1, and corresponding features
        true_labels = np.array(nodes['label'])
        features   = nodes.loc[:,'feature_0':].as_matrix()

        return true_labels, features
