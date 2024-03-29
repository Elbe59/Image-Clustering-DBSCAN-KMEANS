import os
import time as t

import cv2
import matplotlib.colors
import numpy as np
import matplotlib.image as img
from PIL import Image



# Defining our function
def kmeans(image, k, nb_iterations, method):
    global dico_dist
    clusters = {}
    for i in range(k):  # Initialise le disctionnaire qui va contenir les différents clusters
        clusters[i] = []
    idx = np.random.choice(len(image), k, replace=False)  # choisi k index aléatoire dans l'image pour placer les premiers centroides
    centroids = image[idx, :]  # Choisi les premiers centroides de manière aléatoire
    # Calcul de la distance entre chaque labels et les différents centroid
    list_distances = []
    for index in range(len(image)):
        thisDist = []
        for i in range(len(centroids)):
            dist = 0
            if (method == "manhattan"):
                dist = manhattan_distance(image[index], centroids[i])  # type de distance utilisée
            elif (method == "euclidean"):
                dist = euclidean_distance(image[index], centroids[i])  # type de distance utilisée
            else:
                print("Erreur je ne connais pas ce type de distance")
                return 1
            thisDist.append(dist)
        list_distances.append(thisDist)
    #Choisi le centroid avec le minimum de distance
    labels = np.array([np.argmin(i) for i in list_distances])

    # On répète les étapes précédentes jusqu'à atteindre le nombre d'itération désiré
    for i in range(nb_iterations):
        print("Itération: "+str(i+1)+" / "+str(nb_iterations))
        centroids = []
        for idx in range(k):
            # Mise à jour du centroid par la moyenne de son cluster
            if (len(image[labels == idx]) != 0):
                temp_cent = image[labels == idx].mean(axis=0)
                centroids.append(temp_cent)
            else:
                centroids.append([0, 0, 0]) # Si aucun pixel n'était dans le proche du centroid alors le centroid reste avec la couleur (0,0,0)
        centroids = np.vstack(centroids)  #Mise à jours des centroides
        list_distances = []
        for index in range(len(image)):
            thisDist = []
            for i in range(len(centroids)):
                dist = 0
                if (method == "manhattan"):
                    dist = manhattan_distance(image[index], centroids[i])  # type de distance utilisée
                elif (method == "euclidean"):
                    dist = euclidean_distance(image[index], centroids[i])  # type de distance utilisée
                else:
                    print("Erreur je ne connais pas ce type de distance")
                    return 1
                thisDist.append(dist)
            list_distances.append(thisDist)
        labels = np.array([np.argmin(i) for i in list_distances])

    for i in range(len(image)):
        clusters[labels[i]].append(image[i])  # On ajoute chaque labels à son cluster correspondant dans le dictionnaire

    return labels, clusters, centroids  # Retourne la liste des labels, des clusters et les centroids finaux


def save_results(new_img, image_name, k, nb_iterations, method_distance):
    """
    Description :
    Méthode pour sauvegarder les images obtenues dans une répertoire ./results
    """
    repo = './output/K_MEANS/' + image_name + '/'
    if not os.path.exists(repo):
        os.makedirs(repo)

    path = repo + 'res_k=' + str(k) + '_iter=' + str(nb_iterations) + '_dist=' + method_distance[0] + '__' + image_name
    img.imsave(path, new_img)
    return path


def img_load(img_path):
    """
    Description :
    Méthode pour le chargement de l'image.
    :param img_name:
    """
    global width, height
    image = img.imread(img_path);
    width, height, _ = image.shape
    basename = os.path.basename(img_path)
    return image, basename


def euclidean_distance(color1, color2):
    """
    Retourne la distance euclidienne entre deux couleurs
    """
    dist = np.linalg.norm(color1 - color2)
    return dist


def manhattan_distance(color1, color2):
    """
    Retourne la distance de Manhattan entre deux couleurs
    :param color1:
    :param color2:
    :return:
    """
    vector = zip(color1, color2)
    dist = 0
    for val1, val2 in vector:
        dist += abs(int(val1) - int(val2))
    return dist


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def launch(k, nb_iterations, method, image_path, new_width, new_height):
    global image
    start = t.time()
    image, img_name = img_load(image_path)
    print("Programme lancé, pour l'image : "+img_name)

    # reshape the image to be a list of pixels
    image = image_resize(image, width=new_width, height=new_height)
    width, height, _ = image.shape
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    labels, clusters, centroids = kmeans(image=image, k=k, nb_iterations=nb_iterations, method=method)
    identified_palette = centroids  # On récupère les différents centroides

    recolored_img = np.copy(image)
    # Reconstruction de l'image pour l'affichage
    for index in range(
            len(recolored_img)):  # Pour chaque pixel dans l'image à retourner, on regarde à quelle centroid elle correspond
        recolored_img[index] = identified_palette[labels[index]]
    recolored_img = recolored_img.reshape(width, height, 3)
    path = save_results(new_img=recolored_img, image_name=img_name, k=k, nb_iterations=nb_iterations,
                        method_distance=method)
    end = t.time()
    print("Processus terminé en "+str(round(end - start))+" secondes")
    return recolored_img, path, round(end - start)

