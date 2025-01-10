from mixed_dataset import mix_datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class Cluster:
    def __init__(self, n_clusters=2, n_init=10):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        self.scalar = StandardScaler()
        self.cluster_centers = None

    def features(self, state):
        state = state.flatten().reshape(state.shape[0], -1)
        return state
    
    def fit(self, features):
        scaled_features = scaled_features = self.scalar.fit_transform(features)
        self.kmeans.fit(scaled_features)
        self.cluster_centers = self.kmeans.cluster_centers_

    def predict(self, features):
        if self.cluster_centers is None: # check if cluster centers are initialized
            raise ValueError("Cluster centeres not initialized. Run fit first")
                  
        scaled_features = self.scalar.transform(features)
        prediction = self.kmeans.predict(scaled_features)
        return prediction


dataset = mix_datasets("walker2d-medium-v2", "walker2d-expert-v2")

clustering = Cluster(n_clusters=50)

features = clustering.features(dataset['observations'])
clustering.fit(features)
cluster_labels = clustering.predict(features)

counts = Counter(cluster_labels)
print("Cluster Sizes:", dict(counts))

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(features)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
#                 c=cluster_labels, cmap='viridis', s=10)
# plt.colorbar(sc)
# plt.title('PCA 3D with Clusters')
# plt.show()

