import os
import time as t

import cv2

width = 0
height = 0
list_centroid = []
clusters = {}
# Loading the required modules
DIM_IMG = (100,100)
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Defining our function
def kmeans(img, k, no_of_iterations):
    clusters = {}
    for i in range(k): # Initialise le disctionnaire qui va contenir les différents clusters
        clusters[i] = []
    idx = np.random.choice(len(img), k, replace=False) # choisi k index aléatoire dans l'image
    centroids = img[idx, :]  # Choisi les premiers centroides de manière aléatoire

    # finding the distance between centroids and all the data labels
    #distances = cdist(x, centroids, 'euclidean')  # Step 2
    #Calcul de la distance entre chaque labels et les différents centroid
    list_distances = []
    for index in range(len(image)):
        thisDist = []
        for i in range(len(centroids)):
            dist = euclidean_distance(image[index],centroids[i]) # type de distance utilisée
            thisDist.append(dist)
        list_distances.append(thisDist)
    # Choisi le centroid avec le minimum de distance
    labels = np.array([np.argmin(i) for i in list_distances])  # Step 3

    # On répète les étapes précédentes jusqu'à atteindre le nombre d'itération désirée
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Mise à jour du centroid par la moyenne de son cluster
            if(len(img[labels == idx]) != 0):
                temp_cent = img[labels == idx].mean(axis=0)
                centroids.append(temp_cent)
            else:
                centroids.append([0,0,0])
        centroids = np.vstack(centroids)  # Updated Centroids

        #distances = cdist(x, centroids, 'euclidean')
        list_distances = []
        for index in range(len(image)):
            thisDist = []
            for i in range(len(centroids)):
                dist = euclidean_distance(image[index], centroids[i])
                thisDist.append(dist)
            list_distances.append(thisDist)
        labels = np.array([np.argmin(i) for i in list_distances])

    for i in range(len(image)):
        clusters[labels[i]].append(image[i])  # On ajoute chaque labels à son cluster correspondant dans le dictionnaire

    return labels, clusters, centroids # Retourne la liste des labels, des clusters et les centroids finaux


def centeroidnp(cluster):
    global list_centroid
    length = cluster.shape[0]
    sum_r = np.sum(cluster[:, 0])
    sum_g = np.sum(cluster[:, 1])
    sum_b = np.sum(cluster[:, 2])
    if (length != 0):
        centroid = np.array([round(sum_r / length), round(sum_g / length), round(sum_b / length)])
        list_centroid.append(centroid)
        return centroid
    return image[0]


def save_results(image, img_name, nb_cluster):
    """
    Description :
    Méthode pour sauvegarder les images obtenues dans une répertoire ./results
    """
    if not os.path.exists('output/'):
        os.makedirs('output/')

    path = './output/RESULT-K=' + str(nb_cluster) + '-' + img_name
    cv2.imwrite(path, image)
    return path


def img_load():
    """
    Description :
    Méthode pour le chargement de l'image.
    """
    global width, height
    image = cv2.imread('./images/maison.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width, height, _ = image.shape
#    img_ref = cv2.resize(img_ref, DIM_IMG)
    print(image.shape)
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    plt.show()
    return image


def launch1():
    image = img_load()
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    print(image)
    nb_cluster = 16
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
    save_results(image, 'le-roi-lion.jpg', nb_cluster)


# def euclidean_distance(color1, color2):
#     """
#     Retourne la distance euclidienne entre deux couleurs
#     """
#     # dist = np.linalg.norm(color1 - color2)
#     dist = np.sqrt(np.sum((color1 - color2) ** 2))
#     return dist


def recalculate_clusters(image, centroids, k):
    global clusters
    for pixel in image:
        list_euc_dist = []
        for j in range(k):
            list_euc_dist.append(euclidean_distance(pixel, centroids[j]))
        clusters[list_euc_dist.index(min(list_euc_dist))].append(pixel)
    return clusters


def recalculate_centroids(centroids, clusters, k):
    new_centroids = {}
    for i in range(k):
        new_centroids[i] = centeroidnp(clusters[i])
    return new_centroids
def euclidean_distance(color1,color2):
    """
    Retourne la distance euclidienne entre deux couleurs
    """
    dist = np.linalg.norm(color1 - color2)
    return dist

def manhattan_distance(color1, color2):
    vector = zip(color1,color2)
    dist = 0
    for val1,val2 in vector:
        dist += abs(int(val1) - int(val2))
    return dist
    #return sum(abs(val1-val2) for val1, val2 in zip(color1,color2))
def kmeansV(image,k,nb_iterations):
    clusters = {}
    for i in range(k):
        clusters[i] = []
    indexes = np.random.choice(len(image),k,replace=False)
    centroids = image[indexes, :] # Sélection aléatoire des centroids
    print(centroids)
    list_distances = []
    for index in range(len(image)):
        dist = euclidean_distance(image[index],centroids)
        list_distances.append(dist)
    print(min(list_distances))
    #recherche du centroid avec la plus petite distance:
    points = np.array([np.argmin(i) for i in list_distances])
    print(points)
    for _ in range(nb_iterations):
        centroids = []
        for indexes in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = image[points == indexes].mean(axis=0)
            centroids.append(temp_cent)
        centroids = np.vstack(centroids)  # Updated Centroids
        list_distances = []
        for index in range(len(image)):
            dist = euclidean_distance(image[index], centroids)
            list_distances.append(dist)
        print(min(list_distances))
        # recherche du centroid avec la plus petite distance:
        points = np.array([np.argmin(i) for i in list_distances])


    return points


def test():
    k = 5
    global clusters
    centroids = {}
    for i in range(k):
        clusters[i] = []
    for i in range(k):
        print(np.array(image[i]))
        centroids[i] = np.array(image[i])

    new_centroids = centroids
    start = True
    while new_centroids != centroids or start == True:
        start = False
        clusters = recalculate_clusters(image, new_centroids, k)
        new_centroids = recalculate_centroids(centroids, clusters, k)
        print(centroids)
        print(new_centroids)
    print(clusters)
    print(centroids)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
if __name__ == '__main__':
    start = t.time()
    global image
    image = img_load()
    # reshape the image to be a list of pixels
    image = image_resize(image,width=500)
    width,height,_ = image.shape
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    labels, clusters, centroids = kmeans(image, 32, 10)
    identified_palette = centroids # On récupère les différents centroides

    recolored_img = np.copy(image)
    for index in range(len(recolored_img)): # Pour chaque pixel dans l'image à retourner, on regarde à quelle centroid elle correspond
        recolored_img[index] = identified_palette[labels[index]]
    #Reconstruction de l'image pour l'affichage
    recolored_img = recolored_img.reshape(width,height, 3)
    plt.figure()
    plt.axis("off")
    plt.imshow(recolored_img)
    plt.show()
    end = t.time()
    print(str(end-start) + " secondes")