from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_inertias_for_clustering(df):
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)
    
    return inertias


def plot_inertia_k(df, inertias):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    plt.show()
    
    
def cluster_using_kmeans(df):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df)
    label = kmeans.predict(df)
    df['label'] = label
    print(df)
    return df

def visualize_clusters(df):
    df['label'].plot(kind='hist')
    plt.plot(df['Age'], df['Annual Income (k$)'], df['label'], 'go')
    for label in df['label']:
        subset = df[df['label'] == label]
        if (label == 0):
            plt.scatter(subset['Age'], subset['Annual Income (k$)'], s=100, c="red")
        elif label == 1:
            plt.scatter(subset['Age'], subset['Annual Income (k$)'], s=100, c="blue")
        elif label == 2:
            plt.scatter(subset['Age'], subset['Annual Income (k$)'], s=100, c="green")