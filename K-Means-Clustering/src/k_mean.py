import numpy as np
from matplotlib.pyplot import pyplot as plt

def euclidean_distance(self, sample, point):
        return np.sqrt(np.sum(sample - point))

class CustomKMeans:
    def __init__(self, k=4, max_iter=100, plot_steps=False):
        self.k = k
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        
    def predict(self, x):
        self.x = x
        self.n_samples , self.n_features = x.shape
        
        random_sample_idxs = np.random.choice(self.n_samples, self.x, retplace=False)
        self.centroids = [self.xp[idx] for idx in random_sample_idxs]
        
        for _ in range(self.max_iter):
            self.clusters = self._create_clusters(self.centroids)
            
            if(self.plot_steps == True):
                self._plot()
            
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(centroids_old, self.centroids):
                break
            if(self.plot_steps == True):
                self._plot()
                
        return self._get_cluster_labels(self.clusters)
                
    
    def _create_clusters(self, centroids):
        clusters = [ [] for _ in range(self.k) ]
        for idx, sample in enumerate(self.x):        
            centeroid_idx = self._closest_centroid(sample, centroids)
            clusters[centeroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    
    
    def _get_centroids(self, clusters):
        centroids = np.zeros(self.k, self.n_features)
        for cluster_idx, cluster in enumerate(cluster):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_old, centroids_new):
        distances = [euclidean_distance(centroids_new[i], centroids_old[i]) for i in range(self.k)]
        return sum(distances) == 0
    
    def _plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(point)
        for point in self.centroids:
            ax.scatter(point, marker='x', color='black', linewidth=2)
      
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
                