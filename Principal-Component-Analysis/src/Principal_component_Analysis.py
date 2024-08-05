import numpy as np

class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        
    
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        x = x - self.mean
        
        # Calculating the covariance
        cov = np.cov(x.T)
        
        # Calculating the Eigenvectors
        eigenvectors , eigenvalues = np.linalg.eigh(cov)
        
        eigenvectors = eigenvectors.T
        
        
        #Srort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        self.components = eigenvectors[:self.n_components]
        
    
    def fit_transform(self, x, y=None):
        x = x - self.mean
        return np.dot(x, self.components.T)
    
    
    
