from sklearn.neighbors  import KNeighborsClassifier
from k_nearest_neighbors import CustomKNN
from sklearn.model_selection import train_test_split


def train_model(x_train, y_train, k, learning_rate, model_type='custom'):
    if model_type == 'custom':
        model = CustomKNN(k)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        return model





