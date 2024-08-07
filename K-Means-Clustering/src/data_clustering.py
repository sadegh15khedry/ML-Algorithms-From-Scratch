from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from k_mean import CustomKMeans

def get_cluster_model(df, k, model_type='custom'):
    if model_type == 'sklearn':
        model = KMeans(n_clusters=k)
        model.fit(df)
        return model
    elif model_type == 'custom':
        model = CustomKMeans(k)
        model.fit(df)
        return model

def get_inertias_for_clustering(df):
    inertias = []
    for k in range(1, 11):
        kmeans = get_cluster_model(df, k, 'sklearn')
        inertias.append(kmeans.inertia_)
    
    return inertias


def plot_inertia_k(inertias):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('../results/inertia.png')
    plt.show()
    
def visualize_clusters(df):
    df['label'].plot(kind='hist')
    plt.plot(df['age'], df['annual_income'], df['label'], 'go')
    for label in df['label']:
        subset = df[df['label'] == label]
        if (label == 0):
            plt.scatter(subset['age'], subset['annual_income'], s=100, c="red")
        elif label == 1:
            plt.scatter(subset['age'], subset['annual_income'], s=100, c="blue")
        elif label == 2:
            plt.scatter(subset['age'], subset['annual_income'], s=100, c="green")
        elif label == 3:
            plt.scatter(subset['age'], subset['annual_income'], s=100, c="yellow")


