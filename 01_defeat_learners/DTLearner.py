"""A simple wrapper for Decision Tree regression"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from copy import deepcopy


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False, tree=None):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = deepcopy(tree)
        if verbose:
            self.get_learner_info()
        

    def __build_tree(self, dataX, dataY):
        """Builds the Decision Tree recursively by choosing the best feature to split on and 
        the splitting value. The best feature has the highest absolute correlation with dataY. 
        If all features have the same absolute correlation, choose the first feature. 
        The splitting value is the median of the data according to the best feature

        Parameters:
        dataX: An ndarray of X values at each node
        dataY: A 1D array of Y training values at each node
        
        Returns:
        tree: A decision tree in the form of an ndarray
        """
        # Get the number of samples (rows) and features (columns) of dataX
        num_samples = dataX.shape[0]
        num_feats = dataX.shape[1]

        # If there are <= leaf_size samples or all data in dataY are the same, return leaf
        if num_samples <= self.leaf_size or len(pd.unique(dataY)) == 1:
            return np.array([-1, dataY.mean(), np.nan, np.nan])
        else:
            # Initialize best feature index and best feature correlation
            best_feat_i = 0
            best_abs_corr = 0.0

            # Determine best feature to split on, using correlation between features and dataY
            for feat_i in range(num_feats):
                abs_corr = abs(pearsonr(dataX[:, feat_i], dataY)[0])
                if abs_corr > best_abs_corr:
                    best_abs_corr = abs_corr
                    best_feat_i = feat_i
            
            # Split the data according to the best feature
            split_val = np.median(dataX[:, best_feat_i])

            # Logical arrays for indexing
            left_index = dataX[:, best_feat_i] <= split_val
            right_index = dataX[:, best_feat_i] > split_val

            # Build left and right branches and the root
            lefttree = self.__build_tree(dataX[left_index], dataY[left_index])
            righttree = self.__build_tree(dataX[right_index], dataY[right_index])

            # Set the starting row for the right subtree of the current root
            if lefttree.ndim == 1:
                righttree_start = 2 # The right subtree starts 2 rows down
            elif lefttree.ndim > 1:
                righttree_start = lefttree.shape[0] + 1
            root = np.array([best_feat_i, split_val, 1, righttree_start])

            return np.vstack((root, lefttree, righttree))
        

    def addEvidence(self, dataX, dataY):
        pass
        
        
    def query(self, points):
        """Estimate a set of test points given the model we built
        
        Parameters:
        points: A numpy array with each row corresponding to a specific query
        Returns: the estimated values according to the saved model
        """
        pass


    def get_learner_info(self):
        print ("Info about this Decision Tree Learner:")
        print ("leaf_size =", self.leaf_size)
        if self.tree is not None:
            print ("tree shape =", self.tree.shape)
            print ("tree as a matrix: \n", self.tree)
            # Create a dataframe from tree for a user-friendly view
            df_tree = pd.DataFrame(self.tree, columns=["factor", "split_val", "left", "right"])
            df_tree.index.name = "node"
            print (df_tree)
        else:
            print ("Tree has no data")


if __name__=="__main__":
    print ("This is a Decision Tree Learner")