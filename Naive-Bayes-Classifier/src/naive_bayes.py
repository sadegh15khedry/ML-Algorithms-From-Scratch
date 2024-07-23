import numpy as np

class CustomNaiveBayesClassifier:
    def __init__(self):
        print("Custom Naive Bayes Classifier")
        
        
        
    def fit(self, x, y):
        number_of_samples, number_of_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # Calculating the mean, variance, and prior for each class
        self.mean = np.zeros((n_classes, number_of_features), dtype=np.float64)
        self.var = np.zeros((n_classes, number_of_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)
        
        for index, c in enumerate(self._classes):
            x_class = x[y == c]
            self.mean[index, :] = x_class.mean(axis=0)
            self.var[index, :] = x_class.var(axis=0)
            self._priors[index, :] = x_class.shape[0] / float(number_of_samples)
            
                
    def predict(self, x):
        y_pred = [self._predict(i) for i in x]
        return np.array(y_pred)
    
    def _predict(self, x):
        