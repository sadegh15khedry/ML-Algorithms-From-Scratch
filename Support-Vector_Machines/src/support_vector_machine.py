import numpy as np

class CustomSupportVectorMachine:
    def __init__(self,learning_rate=0.001, lambda_param=0.01, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, x, y):
        x = x.astype(float)
        y = y.astype(float)

        n_samples, n_features = x.shape
        
        y_encoded = y.replace({0: -1, 1: 1})
        y_encoded = y_encoded.to_numpy()
        x = x.to_numpy()

        # y_ = np.where(y <= 0, -1, 1)
        # print(type(y_encoded))
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.number_of_iterations):
            for idx, x_i in enumerate(x):
                value = y_encoded[idx] * (np.dot(x, self.weights) - self.bias)
                condition = value[0] >= 1
                if bool(condition):
                    self.weights -= self.learning_rate * (2 * self.lambda_param - np.dot(x_i, y_encoded[idx]))
                    self.bias -= self.learning_rate * y_encoded[idx]
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
     
                    
                    
                     
             
    def predict(self, x):
        x = x.astype(float)
        approx = np.dot(x,self.weights) - self.bias
        return np.sign(approx)

