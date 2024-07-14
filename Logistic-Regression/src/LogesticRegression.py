import numpy as np

def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result

class LogesticRegression():
    
    def __init__(self, learning_rate, number_of_iterations):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None
        
        
    def fit(self, x, y):
        number_of_samples, number_of_features = x.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0
        
        for i in range(self.number_of_iterations):
            linear_prdictions = np.dot(self.weights, x) + self.bias
            predictions = sigmoid(linear_prdictions)

            dw = (1/number_of_samples) * np.dot(x.T, (predictions, predictions - y))
            db = (1/number_of_samples) * np.sum(predictions -y)
            
            self.weights = self.weights - (dw * self.learning_rate)
            self.bias = self.bias - (db * self.learning_rate)


    def predict(self, x):
        linear_prdictions = np.dot(self.weights, x) + self.bias
        y_pred = sigmoid(linear_prdictions)
        
        class_pred = []
        for y in y_pred:
            if y <= .5:
                class_pred.append(0)
            else: 
                class_pred.append(1)
                
        return class_pred
                
        

        
        
    