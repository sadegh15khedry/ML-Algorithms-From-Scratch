import numpy as np
from node import Node
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class CustomDecisionTree():
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = max_depth
        self.root = None
        self.label_encoder = LabelEncoder()
    
    def fit(self, x, y):
        # Ensure y is encoded
        y_encoded = self.label_encoder.fit_transform(y)
        self.n_features = x.shape[1] if self.n_features is None else min(self.n_features, x.shape[1])
        self.root = self._grow_tree(x.to_numpy(), y_encoded)

    
    


    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))
        
        # Check the stoping criteria
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best Split
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold  = self._best_split(x, y, feature_indices)
        
        left_idxs, right_idxs = self._split(x[:, best_feature],best_threshold,)
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    
    def _best_split(self, x, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None
        
        for feature_index in feature_indices:
            x_column = x[:, feature_index]  # Use NumPy indexing
            thresholds = np.unique(x_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, x_column, threshold)
                
                # Ensure gain is a scalar value and compare
                if isinstance(gain, (int, float)) and gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold
        return split_index, split_threshold
    
    
    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _information_gain(self, y, x_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_idx, right_idx = self._split(x_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0  # Avoid division by zero or empty splits
        
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx) 
        
        # Ensure y[left_idx] and y[right_idx] are arrays
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    
    def _split(self, x_column, threshold):
        left_idx = np.where(x_column <= threshold)[0]
        right_idx = np.where(x_column > threshold)[0]
        return left_idx, right_idx
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum(p * np.log(p) for p in ps if p > 0)
    
    
    def predict(self, x):
        result = []
        print(type(x))
        for _, row in x.iterrows():  # Use .iterrows() to iterate over DataFrame rows
            result.append(self._traverse_tree(row, self.root))
        return np.array(result)
        
    
    
    # def _traverse_tree(self, x, node):        
    #     if node.is_leaf_node():
    #         return node.value()
    #     elif x[node.feature] <= node.threshold:
    #         return self._traverse_tree(x, node.left)
    #     return self._traverse_tree(x, node.right)

    def _traverse_tree(self, x, node):
        print(node.threshold)  # Debugging print statements
        print(x)               # Debugging print statements
        if node.is_leaf_node():
            return node.value  # Directly access value, do not use parentheses
        elif x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

        
        