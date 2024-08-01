from decision_tree import CustomDecisionTree
import numpy as np
from collections import Counter

class CustomRandomForest:
    
    def __init__(self, x, y, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
        
    def fit(self,x, y):
        self.n_features = x.shape[1]
        for _ in range(self.n_trees):
            model = CustomRandomForest(max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       n_features=self.n_features)
            x_sample, y_sample = get_sample_data(x, y, 0.2)
            model.fit(x_sample, y_sample)
            self.trees.append(model)
            
    def predict(self, x):
        results = []
        for row in x.shape[0]:
            labels = []
            for model_index in range(self.n_trees):
                model = self.trees[model_index]
                label = model.predict(row)
                labels.append(label)
            most_common = self._most_common(labels)
            results.append(most_common)
        results = np.array(results)
        return results
                
    
    def get_sample_data(x, y, percentage=0.2):
        number_of_samples = x.shape[0] * percentage
        indx = np.random.choice(number_of_samples, number_of_samples, replace=True)
        return x[indx], y[indx]
    
    def _most_common(labels):
        counter = Counter(labels)
        most_common = counter.most_common(1)
        return most_common
    
