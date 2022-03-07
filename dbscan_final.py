import os
import random
import time as t
import cv2
import numpy as np
import pandas as pd

width = 0
height = 0
clusters = []
list_centroid = []


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


def save_results(new_img, image_name, eps, minPts, method_distance):
    """
    Description :
    Méthode pour sauvegarder les images obtenues dans une répertoire ./results
    """
    repo = './output/DBSCAN/' + image_name + '/'
    if not os.path.exists(repo):
        os.makedirs(repo)

    path = repo + 'res_eps=' + str(eps) + '_minPts=' + str(minPts) + '_dist=' + method_distance[0] + '__' + image_name
    image = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)  # Il faut retransformer en type bgr pour openCV
    cv2.imwrite(path, image)
    return path


def img_load(img_path):
    """
    Description :
    Méthode pour le chargement de l'image.
    :param img_name:
    """
    global width, height
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width, height, _ = image.shape
    basename = os.path.basename(img_path)
    return image,basename


def type_point(eps, minimumPts, df, index, method_distance):
    # Récupère le vecteur rgb
    if (method_distance != "manhattan" and method_distance != "euclidean"):
        print("La distance demandée n'est pas reconnu")
        return 1
    r, g, b = df.iloc[index]["R"], df.iloc[index]["G"], df.iloc[index]["B"]
    temp = []
    # temp va contenir un nouveau dataframe avec tous les points respectant le critère de distance inférieur à epsilon
    if(method_distance == "euclidean"):
        temp = df[(((r - df['R']) ** 2 + (g - df['G']) ** 2 + (b - df['B']) ** 2)**(1/2) <= eps) & (df.index != index)]
    if(method_distance == "manhattan"):
        temp = df[((np.abs(r - df['R']) + np.abs(g - df['G']) + np.abs(b - df['B'])) <= eps) & (df.index != index)]
    if (len(temp) >= minimumPts):
        return (temp.index, True, False, False)
    elif (len(temp) < minimumPts and len(temp) > 0):
        return (temp.index, False, True, False)
    elif (len(temp) == 0):
        return (temp.index, False, False, True)


def predict_clusters(eps, minPts, method_distance, df):
    label_clusters = 1
    labels = np.full(shape=df.shape[0], fill_value=0)
    stack_c = set()
    visiteNo = list(df.index)
    clusters = []
    list_clusters = {}
    list_clusters[0] = []
    list_clusters[1] = []
    while (len(visiteNo) != 0):
        first_pt = True  # On identifie que le point actuel est le premier point du cluster
        stack_c.add(random.choice(visiteNo))  # On choisi un point non visité
        while (
                len(stack_c) != 0):  # Tant que la pile n'est pas vide c'est qu'il y a encore des points à traiter dans le cluster
            c_index = stack_c.pop()  # Dernier point de la pile
            # Verifie si le point à l'index c_index est un coeur, une bordure ou bien noise
            indexVoisins, coeur, bordure, noise = type_point(eps, minPts, df, c_index, method_distance)
            if (bordure and first_pt):
                clusters.append((c_index, 0))  # Cluster label 0 pour les points isolé
                labels[c_index] = 0
                list_clusters[0].append(c_index)
                clusters.extend(list(zip(indexVoisins, [0 for _ in range(len(indexVoisins))])))
                visiteNo.remove(c_index)
                visiteNo = [e for e in visiteNo if e not in indexVoisins]
                continue
            visiteNo.remove(c_index)
            indexVoisins = set(indexVoisins) & set(visiteNo) # On garde uniquement les voisins non visité
            if (coeur):
                first_pt = False
                clusters.append((c_index, label_clusters))
                labels[c_index] = label_clusters
                list_clusters[label_clusters].append(c_index)
                stack_c.update(indexVoisins)
            elif (bordure):
                clusters.append((c_index, label_clusters))
                list_clusters[label_clusters].append(c_index)
                labels[c_index] = label_clusters
                continue
            elif (noise):
                clusters.append((c_index, 0))
                list_clusters[0].append(c_index)
                labels[c_index] = 0
                continue
        if not first_pt:
            label_clusters += 1
            list_clusters[label_clusters] = []
    return labels, list_clusters



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

def launch(eps, minPts, method, image_path,new_width,new_height):
    global image
    start = t.time()
    image,image_name = img_load(img_path=image_path)
    image = image_resize(image,width=new_width,height=new_height)
    width,height,_ = image.shape
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    data = pd.DataFrame(image, columns=["R", "G", "B"]).astype(np.int64)
    labels, list_clusters = predict_clusters(eps=eps, minPts=minPts, method_distance=method, df=data)
    for key in list_clusters.keys():
        centeroidnp(image[list_clusters[key]])
    # Palette de couleur à partir des centroides
    identified_palette = np.array(list_centroid).astype(int)
    # recolor the entire image
    recolored_img = np.copy(image)
    for index in range(len(recolored_img)):
        if (labels[index] == 0):  # Il faut prendre en compte le dernier label qui correspond aux pixels ne faisant partie d'aucun cluster
            recolored_img[index] = np.array([255, 255, 255])
        else:
            recolored_img[index] = identified_palette[labels[index]]
    recolored_img = recolored_img.reshape(width, height, 3)
    path = save_results(recolored_img,image_name,eps,minPts,method)

    end = t.time()
    return recolored_img,path,round(end-start)
