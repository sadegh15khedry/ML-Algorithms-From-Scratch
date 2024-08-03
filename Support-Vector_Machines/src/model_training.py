from sklearn import svm

from support_vector_machine import CustomSupportVectorMachine 


def train_model(x_train, y_train, number_of_iterations, learning_rate, model_type='custom'):
    if model_type == 'custom':
        model = CustomSupportVectorMachine(learning_rate, number_of_iterations)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = svm.SVC()
        model.fit(x_train, y_train)
        return model




