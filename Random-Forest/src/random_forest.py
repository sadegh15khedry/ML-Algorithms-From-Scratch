from decision_tree import CustomDecisionTree
import numpy as np
from collections import Counter

class CustomRandomForest:
    
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        """
        Initialize the Random Forest classifier.

        Parameters:
        - n_trees: Number of trees in the forest.
        - max_depth: Maximum depth of each tree.
        - min_samples_split: Minimum number of samples required to split an internal node.
        - n_features: Number of features to consider when looking for the best split.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
        
    def fit(self,x, y):
        """
        Train the Random Forest classifier.

        Parameters:
        - x: Training features.
        - y: Training labels.
        """
        self.n_features = x.shape[1]
        for _ in range(self.n_trees):
            model = CustomDecisionTree(max_depth=self.max_depth,
                                       min_samples_split=self.min_samples_split,
                                       n_features=self.n_features)
            x_sample, y_sample = self._get_sample_data(x, y, 0.2)
            model.fit(x_sample, y_sample)
            self.trees.append(model)
            
    def predict(self, x):
        print(x.shape)
        """
        Predict the class labels for the provided samples.

        Parameters:
        - x: Test features.

        Returns:
        - numpy array of predicted labels.
        """
        results = []
        # print(type(x))
        # for index, row in x.iterrows():
        #     print(type(row))
        #     print(row)
            # row = np.array(row)
        # labels = []
        # for model_index in range(self.n_trees):
        #     model = self.trees[model_index]
        #     label = model.predict(x)
        #     labels.append(label)
            
        #     most_common = self._most_common(labels)
        #     results.append(most_common)
        # results = np.array(results)
        # return results
        predictions = np.array([tree.predict(x) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common(pred) for pred in predictions])
        return predictions
                
    
    def _get_sample_data(self, x, y, percentage=0.2):
        """
        Randomly sample the data for bootstrapping.

        Parameters:
        - x: Features.
        - y: Labels.
        - percentage: Proportion of data to sample.

        Returns:
        - Tuple of sampled features and labels.
        """
        number_of_samples = int(x.shape[0] * percentage)
        indx = np.random.choice(x.shape[0], number_of_samples, replace=True)
        return x.iloc[indx], y.iloc[indx]
    
    def _most_common(self, labels):
        """
        Find the most common label.

        Parameters:
        - labels: List of labels.

        Returns:
        - The most common label.
        """
        counter = Counter(labels)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
