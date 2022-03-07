import os
import random
import time as t

import cv2
import matplotlib.pyplot as plt
import numpy as np

width = 0
height = 0

epsilon = 8
minimum_points = 1
visiteOk = []
voisin = {}
clusters = []
list_centroid = []

def centeroidnp(cluster):
    global list_centroid
    length = cluster.shape[0]
    sum_r = np.sum(cluster[:, 0])
    sum_g = np.sum(cluster[:, 1])
    sum_b = np.sum(cluster[:, 2])
    centroid = np.array([round(sum_r/length),round(sum_g/length),round(sum_b/length)])
    list_centroid.append(centroid)
    return centroid

def get_centroids():
    nb_centroid = len(clusters)
    # creating an empty centroid array
    Centroids = np.array([]).reshape(1, 0)

    for i in range(nb_centroid):
        rand = random.randint(0, len(image) - 1)
        Centroids = np.c_[Centroids, image[rand]]
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
    image = cv2.imread('./images/pumba.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    width,height,_ = image.shape
    #width, height = 100,100
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


def get_voisin(indexPixel):
    """
    Donne la liste d'index des voisins d'un certain pixel
    Il faut que la distance soit plus petite que la valeur d'epsilon pour que 2 pixels soient voisin
    """
    voisinsPixel = []
    indexes = np.arange(len(image))
    for index, _sample in enumerate(image[indexes!=indexPixel]):
        distance = euclidean_distance(image[indexPixel],_sample)
        #print(index in visiteOk)
        # if(index in visiteOk):
        #     print("cc")
        if(distance < epsilon and index not in visiteOk):
            voisinsPixel.append(index)
            #print("cc")

    return np.array(voisinsPixel)


def extension_cluster(indexPixel, listPixelVoisin):
    """
    Parcourt tous les voisins pour déterminer si une expansion du cluster est possible ou non.
    """
    global visiteOk
    global voisin
    cluster = [indexPixel]
    for voisinPixel in listPixelVoisin: #Pour chacun des voisins on regarde si il s'agit également d'un core point
        if voisinPixel not in visiteOk:
            visiteOk.append(voisinPixel) # On ajoute le voisin à la liste des pixels traité
            voisin[voisinPixel] = get_voisin(voisinPixel)
            if(len(voisin[voisinPixel]) >= minimum_points): # Si il s'agit d'un core point on applique le même processus d'expansion de cluster
                expansion = extension_cluster(voisinPixel,voisin[voisinPixel])
                cluster = cluster + expansion
            else: # Si le pixel voisin n'est pas un core point alors on ajoute uniquement ce pixel au cluster
                cluster.append(voisinPixel)
    return cluster

def extension_cluster2(indexPixel, listPixelVoisin):
    """
    Parcourt tous les voisins pour déterminer si une expansion du cluster est possible ou non.
    """
    global visiteOk
    global voisin
    cluster = [indexPixel]
    listPixelVoisin = set()
    for voisinPixel in listPixelVoisin: #Pour chacun des voisins on regarde si il s'agit également d'un core point
        if voisinPixel not in visiteOk:
            visiteOk.append(voisinPixel) # On ajoute le voisin à la liste des pixels traité
            voisin[voisinPixel] = get_voisin(voisinPixel)
            if(len(voisin[voisinPixel]) >= minimum_points): # Si il s'agit d'un core point on applique le même processus d'expansion de cluster
                expansion = extension_cluster(voisinPixel,voisin[voisinPixel])
                cluster = cluster + expansion
            else: # Si le pixel voisin n'est pas un core point alors on ajoute uniquement ce pixel au cluster
                cluster.append(voisinPixel)
    return cluster
def get_cluster_labels():
    """
    Attribue à chaque pixel son cluster ( chaque cluster sera identifié par un label)
    Tous les pixels appartenant à aucun cluster auront le même label
    """
    labels = np.full(shape=image.shape[0],fill_value=len(clusters))
    for cluster_i, cluster in enumerate(clusters):
        for pixelIndex in cluster:
            labels[pixelIndex] = cluster_i
    return labels


def predict(image):
    global visiteOk
    global clusters
    global voisin
    visiteOk = []
    clusters = []
    voisin = {} # Va contenir les voisins de chaque cluster
    nb_pixels = len(image)
    for pixel in range(nb_pixels):
        if(pixel in visiteOk): # Si le pixel a déjà été parcouru, on passe au pixel suivant
            continue
        voisin[pixel] = get_voisin(pixel) #On récupère tous les pixels voisins du pixel rentré en paramètre
        if(len(voisin[pixel]) >= minimum_points): # Si il s'agit d'un "core point"
            visiteOk.append(pixel) # On ajoute le pixel à la liste des pixels déjà visité
            # Il faut maintenant checker un à un les différents voisins pour voir si une extension du cluster est possible
            new_cluster = extension_cluster(pixel,voisin[pixel])
            clusters.append(new_cluster)
    cluster_labels = get_cluster_labels()
    return cluster_labels
if __name__ == '__main__':
    start = t.time()
    global image
    image = img_load()
    #image = cv2.resize(image, (75,75))
    plt.imshow(image)
    plt.show()
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    print(np.shape(image)[0])
    labels = predict(image)
    labels = list(labels)
    print(labels)
    # print(clusters)
    # print(clusters[0])
    # print(image[clusters[0][0]])
    print(clusters)
    for i in range(len(clusters)):
        print(centeroidnp(image[clusters[i]]))

    # the cluster centroids is our color palette
    identified_palette = np.array(list_centroid).astype(int)
    # recolor the entire image
    recolored_img = np.copy(image)
    for index in range(len(recolored_img)):
        if(labels[index] == len(clusters)): # Il faut prendre en compte le dernier label qui correspond aux pixels ne faisant partie d'aucun cluster
            recolored_img[index] = np.array([255,255,255])
        else:
            recolored_img[index] = identified_palette[labels[index]]
    recolored_img = recolored_img.reshape(width, height, 3)
    plt.imshow(recolored_img)
    plt.show()
    end = t.time()
    print("temps: "+str(end-start))

