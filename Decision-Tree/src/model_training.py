from sklearn import tree

from decision_tree import CustomDecisionTree


def train_model(x_train, y_train, model_type='custom'):
    if model_type == 'custom':
        model = CustomDecisionTree()
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = tree.DecisionTreeClassifier()
        model.fit(x_train, y_train)
        return model




