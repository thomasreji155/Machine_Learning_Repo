import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:

    def __init__(self, K=5, max_iters=100, plot_converge=False):
        self.K = K 
        self.max_iters = max_iters
        self.plot_converge = plot_converge

        # list of sample indices from each cluster
        self.clusters = [[] for _ in range(self.K)]
        # The centers (mean feature vector) for each clusters
        self.centroids = []

    
    def predict(self, X):
        self.X = X 
        self.n_sample, self.n_features = X.shape

        #  Initalize the sample clusters
        random_sample_idxs = np.random.choice(self.n_sample, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization to converge the centroids of clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # if self.plot_converge:
            #     self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check if clusters have changed
            if self.is_converged(centroids_old, self.centroids):
                break

            if self.plot_converge:
                self.plot()

        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):

        labels = np.empty(self.n_sample)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        """Create the closest clusters from the given centroid
        """
        # Assign the samples to the closest centroid to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters


    def _closest_centroid(self, sample, centroids):
        """Takes the current sample and check for closest centroid in 
        given centroids.
        """
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index


    def _get_centroids(self, clusters):
        """Gets the new updated centroids
        """
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, centroids_old,  centroids):
        # Distance between each old and new centroids  for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids_old[i]) for i in range(self.K)]
        return sum(distances) == 0


    def plot(self):
        fig,  ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T 
            ax.scatter(*point)


        for point in self.centroids:
            ax.scatter(*point, marker='x', color="black", linewidth=2)

        plt.show()


# using a data set 
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iters=150, plot_converge=True)
y_pred = k.predict(X)

k.plot()       
