import numpy as np

class CustomLinearRegression:
    def __init__(self,learning_rate=0.001, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
        
    def fit(self, x, y):
        number_of_samples, number_of_features = x.shape
        self.weights = np.zeros((number_of_features, 1))
        self.bias = 0
        print(self.number_of_iterations)
        
        x = np.array(x)
        y = np.array(y)
        
        for i in range(self.number_of_iterations):
            y_pred = np.dot(x, self.weights) + self.bias
            
            dw = (1/number_of_samples) * np.dot(x.T, (y_pred - y))
            db = (1/number_of_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Debugging: print the first few values of weights and bias at certain iterations
            if i % 100 == 0:
                print(f"Iteration {i}")
                print(f"First few weights: {self.weights[:5].T}")
                print(f"Bias: {self.bias}")
        
                
    def predict(self, x):
        x = np.array(x)
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred