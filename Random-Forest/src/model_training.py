from sklearn.ensemble import RandomForestClassifier

from decision_tree import CustomDecisionTree


def train_model(x_train, y_train, min_samples_split, max_depth, model_type='custom'):
    if model_type == 'custom':
        model = CustomDecisionTree(min_samples_split=2, max_depth=100, n_features=None)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = RandomForestClassifier(max_depth=2, random_state=0)
        model.fit(x_train, y_train)
        return model




