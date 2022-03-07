import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

width = 0
height = 0

def save_results(image ,img_name, nb_cluster):
    """
    Description :
    Méthode pour sauvegarder les images obtenues dans une répertoire ./results
    """
    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    path = './results/'+'RESULT-K='+nb_cluster+'-'+img_name
    cv2.imwrite(path, image)
    return path

def img_load():
    """
    Description :
    Méthode pour le chargement de l'image.
    """
    global width,height
    image = cv2.imread('./images/le-roi-lion.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    width,height,_ = image.shape
    # img_ref = cv2.resize(img_ref, DIM_IMG)
    print(image.shape)
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    plt.show()
    return image
def euclidean_distance(color1,color2):
    """
    Retourne la distance euclidienne entre deux couleurs
    """
    dist = np.linalg.norm(color1 - color2)
    return dist

def manhattan_distance(color1, color2):
    return sum(abs(val1-val2) for val1, val2 in zip(color1,color2))

def _get_neighbors(self, sample_i):
    """ Return a list of indexes of neighboring samples
    A sample_2 is considered a neighbor of sample_1 if the distance between
    them is smaller than epsilon """
    neighbors = []
    idxs = np.arange(len(self.X))
    for i, _sample in enumerate(self.X[idxs != sample_i]):
        distance = euclidean_distance(self.X[sample_i], _sample)
        if distance < self.eps:
            neighbors.append(i)
    return np.array(neighbors)

def _expand_cluster(self, sample_i, neighbors):
    """ Recursive method which expands the cluster until we have reached the border
    of the dense area (density determined by eps and min_samples) """
    cluster = [sample_i]
    # Iterate through neighbors
    for neighbor_i in neighbors:
        if not neighbor_i in self.visited_samples:
            self.visited_samples.append(neighbor_i)
            # Fetch the sample's distant neighbors (neighbors of neighbor)
            self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
            # Make sure the neighbor's neighbors are more than min_samples
            # (If this is true the neighbor is a core point)
            if len(self.neighbors[neighbor_i]) >= self.min_samples:
                # Expand the cluster from the neighbor
                expanded_cluster = self._expand_cluster(
                    neighbor_i, self.neighbors[neighbor_i])
                # Add expanded cluster to this cluster
                cluster = cluster + expanded_cluster
            else:
                # If the neighbor is not a core point we only add the neighbor point
                cluster.append(neighbor_i)
    return cluster

def _get_cluster_labels(self):
    """ Return the samples labels as the index of the cluster in which they are
    contained """
    # Set default value to number of clusters
    # Will make sure all outliers have same cluster label
    labels = np.full(shape=self.X.shape[0], fill_value=len(self.clusters))
    for cluster_i, cluster in enumerate(self.clusters):
        for sample_i in cluster:
            labels[sample_i] = cluster_i
    return labels

# DBSCAN
def predict(self, X):
    self.X = X
    self.clusters = []
    self.visited_samples = []
    self.neighbors = {}
    n_samples = np.shape(self.X)[0]
    # Iterate through samples and expand clusters from them
    # if they have more neighbors than self.min_samples
    for sample_i in range(n_samples):
        if sample_i in self.visited_samples:
            continue
        self.neighbors[sample_i] = self._get_neighbors(sample_i)
        if len(self.neighbors[sample_i]) >= self.min_samples:
            # If core point => mark as visited
            self.visited_samples.append(sample_i)
            # Sample has more neighbors than self.min_samples => expand
            # cluster from sample
            new_cluster = self._expand_cluster(
                sample_i, self.neighbors[sample_i])
            # Add cluster to list of clusters
            self.clusters.append(new_cluster)

    # Get the resulting cluster labels
    cluster_labels = self._get_cluster_labels()
    return cluster_labels

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length
if __name__ == '__main__':
    image = img_load()
    data = np.asarray( image, dtype="int32" )
    print(data)
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    print(image)
    cluster_labels = predict(self=self,image)
    print(cluster_labels)
    nb_cluster = 10
    epsilon = 1.5 # distance entre 2 points
    number_points = 3 #nombre minimum de noeuds présent pour être considéré comme un cluster.
    # 1 Choose random data points
    # draw a circle around this points and add points in this circle
    kmeans = KMeans(n_clusters=nb_cluster)
    s = kmeans.fit(image)
    print(s)
    labels = kmeans.labels_
    print(labels)
    labels = list(labels)
    centroid = kmeans.cluster_centers_
    print(centroid)
    percent = []
    for i in range(len(centroid)):
        j = labels.count(i)
        j = j / (len(labels))
        percent.append(j)
    print(percent)
    plt.pie(percent, colors=np.array(centroid / 255), labels=np.arange(len(centroid)))
    plt.show()
    new_image_labels = kmeans.predict(image)
    print(new_image_labels)
    # the cluster centroids is our color palette
    identified_palette = np.array(centroid).astype(int)

    # recolor the entire image
    recolored_img = np.copy(image)
    for index in range(len(recolored_img)):
        recolored_img[index] = identified_palette[labels[index]]
    recolored_img = recolored_img.reshape(width, height, 3)
    plt.imshow(recolored_img)
    plt.show()
    save_results(image,'le-roi-lion',nb_cluster)