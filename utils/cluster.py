from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Cluster:
    def __init__(self, n_clusters=8):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.scaler = StandardScaler()
        self.fitted = False 
        self.n_clusters = n_clusters

    def fit(self, dataset):
        """Fits the scaler and KMeans model."""
        data_scaled = self.scaler.fit_transform(dataset)
        self.kmeans.fit(data_scaled)
        self.fitted = True

    def predict(self, state):
         """Predicts the cluster label for a single state or a batch of states."""
         if not self.fitted:
             raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'predict'.")


         state_scaled = self.scaler.transform(state)

         label = self.kmeans.predict(state_scaled)
         return label


    def cluster_labels(self, dataset):
        """Predicts cluster labels for a dataset after fitting the model."""
        if not self.fitted:
            self.fit(dataset) 

        return self.predict(dataset)
    

