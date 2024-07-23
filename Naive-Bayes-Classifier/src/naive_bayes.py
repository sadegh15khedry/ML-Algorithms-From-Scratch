import numpy as np

class CustomNaiveBayesClassifier:
    def __init__(self):
        print("Custom Naive Bayes Classifier")
        
        
        
    def fit(self, x, y):
        # print(type(x))
        number_of_samples, number_of_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self.epsilon = 1e-9
        
        # Calculating the mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, number_of_features), dtype=np.float64)
        self._var = np.zeros((n_classes, number_of_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for index, c in enumerate(self._classes):
            x_class = x[y == c]
            self._mean[index, :] = x_class.mean(axis=0)
            self._var[index, :] = x_class.var(axis=0) + self.epsilon
            self._priors[index] = x_class.shape[0] / float(number_of_samples)
            
                
    def predict(self, x):
        # print(type(x))
        
        y_pred = []
        for i in x:
            print(type(i))
            y_pred.append(self._predict(i))
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        # Calculating posteriors for each class
        for index, c in enumerate(self._classes):
            prior = np.log(self._priors[index])
            # print(type(x))
            pdf = self._probability_density(index, x)
            pdf = np.clip(pdf, a_min=self.epsilon, a_max=None)
            posterior = np.sum(np.log(pdf))
            posterior = posterior + prior
            
            posteriors.append(posterior)
            
        # Returing the class with the highest Posteriors
        return self._classes[np.argmax(posteriors)]
    
    def _probability_density(self, class_index, x):
        mean = self._mean[class_index]
        var = self._var[class_index]
        # print(type(x))
        numerator = np.exp(-((x - mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator