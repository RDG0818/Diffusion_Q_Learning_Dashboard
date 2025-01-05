import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mixed_dataset import mix_datasets

def cluster_dataset(dataset, n_clusters=2, n_init=10):  # n_init for multiple restarts
    # 1. Feature Engineering and Scaling
    states = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards'].reshape(-1, 1)  # Reshape to be compatible with concatenation

    features = np.concatenate([states, actions, rewards], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)


    # 2. K-means Clustering with Custom Initialization (K-means++)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, random_state=0) # Use k-means++ for better initialization
    cluster_labels = kmeans.fit_predict(scaled_features)


    return cluster_labels

def evaluate_clustering(dataset, cluster_labels, n_clusters=2):
    true_labels = dataset['sources']
    from collections import Counter
    cluster_mapping = {i: int(Counter(dataset["sources"][cluster_labels==i]).most_common(1)[0][0]) for i in range(n_clusters)}

    remapped_cluster_labels = [cluster_mapping[label] for label in cluster_labels]

    cm = confusion_matrix(true_labels, remapped_cluster_labels)

    return cm



# Example Usage (assuming you have a dictionary called 'dataset'):
dataset = mix_datasets('walker2d-medium-v2', 'walker2d-expert-v2')
cluster_labels = cluster_dataset(dataset, n_clusters=2)  # 2 clusters for expert vs. non-expert

print(evaluate_clustering(dataset, cluster_labels))