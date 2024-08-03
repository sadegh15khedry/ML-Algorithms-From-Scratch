import numpy as np

class CustomSupportVectorMachine:
    def __init__(self,learning_rate=0.001, lambda_param=0.01, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.number_of_iterations):
            for idx, x_i in enumerate(x):
                condition = y_[idx] * (np.dot(x, self.weights) - self.bias) < 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                    
                    
                    
                     
             
    def predict(self, x):
        approx = np.dot(x,self.weights) - self.bias
        return np.sign(approx)


# import numpy as np

# class CustomSupportVectorMachine:
#     def __init__(self, learning_rate=0.001, lambda_param=0.01, number_of_iterations=1000):
#         self.learning_rate = learning_rate
#         self.lambda_param = lambda_param
#         self.number_of_iterations = number_of_iterations
#         self.weights = None
#         self.bias = None
        
#     def fit(self, x, y):
#         n_samples, n_features = x.shape
#         y_ = np.where(y <= 0, -1, 1)
        
#         self.weights = np.zeros(n_features)
#         self.bias = 0
        
#         for _ in range(self.number_of_iterations):
#             for idx, x_i in enumerate(x):
#                 condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) < 1
#                 if condition:
#                     self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - y_[idx] * x_i)
#                     self.bias -= self.learning_rate * y_[idx]
#                 else:
#                     self.weights -= self.learning_rate * 2 * self.lambda_param * self.weights

#     def predict(self, x):
#         approx = np.dot(x, self.weights) - self.bias
#         return np.sign(approx)