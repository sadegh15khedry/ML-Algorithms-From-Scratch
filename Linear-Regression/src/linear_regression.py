import numpy as np

class CustomLinearRegression:
    def __init__(self,learning_rate=0.001, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
        
    def fit(self, x, y):
        number_of_samples, number_of_features = x.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0
        
        for _ in range(number_of_samples):
            y_pred = np.dot(x, self.weights) + self.bias
            
            dw = (1/number_of_samples) * np.dot(x, y_pred - y)
            db = (1/number_of_samples) * np.sum(y_pred - y)
            
            self.weights =  self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        
                
    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred