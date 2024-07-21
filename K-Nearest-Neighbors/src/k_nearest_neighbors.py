import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


class CustomKNN:
    def __init__(self, k):
        self.k = k
        
        
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        
        
    def predict(self, x):
        prediction = [self._predict(i) for i in x]
        return prediction
    
    
    def _predict(self, x):
        # Computing the distances
        distances = [euclidean_distance(x, i) for i in self.x_train]
        
        # Getting the closest k
        k_indices = np.argsort(distances)[:self.k]
        knn_labels = [self.y_train[i] for i in k_indices]
        
        # Majority voting 
        most_common = Counter(knn_labels).most_common()
        return most_common