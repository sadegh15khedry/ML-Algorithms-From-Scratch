import numpy as np

class CustomNaiveBayesClassifier:
    def __init__(self):
        print("Custom Naive Bayes Classifier")
        
        
        
    def fit(self, x, y):
        number_of_samples, number_of_features = x.shape
        self._classes = np.unique(y)
        
        # Calculating the mean, variance, and prior for each class
        self.mean = np.mean()
        
    def predict(self, x):
        pass