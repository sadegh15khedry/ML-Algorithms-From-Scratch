from sklearn.decomposition import PCA
from principal_component_analysis import CustomPCA


def train_model(df, n_components=2, model_type='custom'):
    if model_type == 'custom':
        model = CustomPCA(n_components=2)
        model.fit(df)
        return model
    
    elif model_type == 'sklearn':
        model = PCA(n_components=2)
        model.fit(df)
        return model




