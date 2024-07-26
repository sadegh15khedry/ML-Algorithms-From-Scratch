import numpy as np
from node import Node
from collections import Counter
class CustomDecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = max_depth
        self.root = None
        
    
    def fit(self, x, y):
        self.n_features = x.shape[1] #maybe more
        self.root = self._grow_tree(x, y)    
    
    
    


    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))
        
        # Check the stoping criteria
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_value(y)
            return Node(value=leaf_value)
        
        # Find the best Split
        feature_indexs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold  = self._best_split(x, y, feature_indexs)
        
        left_idxs, right_idxs = self._split(x[:, best_feature],best_threshold,)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    
    def _best_split(self, x, y, feature_indexs):
        best_gain = -1
        split_index, split_threshold = None, None
        
        for feature_index in feature_indexs:
            x_column = x[:, feature_index]
            theresholds = np.unique(x_column)
            
            for thereshold in theresholds:
                gain = self._information_gain(y, x_column, thereshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = thereshold
        return split_index, split_threshold
    
    
    def _most_common_label(y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _information_gain(self, y, x_column, thereshold):
        parent_entropy = self._entropy(y)
        
        left_idx, right_idx = self._split(x_column, thereshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        n = len(y)
        n_l, n_r= len(left_idx), len(right_idx) 
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    
    def _split(self, x_column, thereshold):
        left_idx = np.argwhere(x_column <= thereshold).flatten()
        right_idx = np.argwhere(x_column > thereshold).flatten()
        return left_idx, right_idx
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return - np.sum(p * np.log(ps) for p in ps if p > 0)
    
    
    
    def predict(self, x):
        return np.array([self._traverse_tree(i, self.root) for i in x])
    
    
    def _traverse_tree(self, x, node):
        if node._is_root():
            return node.value()
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)