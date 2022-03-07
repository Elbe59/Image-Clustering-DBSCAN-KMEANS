import os
import random
import time as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    if (length != 0):
        centroid = np.array([round(sum_r / length), round(sum_g / length), round(sum_b / length)])
        list_centroid.append(centroid)
        return centroid
    return image[0]

# def centeroidnp(cluster):
#     global list_centroid
#     length = len(cluster)
#     sum_r = 0
#     sum_g = 0
#     sum_b = 0
#     for i in range(length):
#         sum_r += cluster[i][0]
#         sum_g += cluster[i][1]
#         sum_b += cluster[i][2]


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



def type_point(eps,minimumPts,df,index):
    # Récupère le vecteur rgb
    r,g,b = df.iloc[index]["R"],df.iloc[index]["G"],df.iloc[index]["B"]
    color_vector = np.array((r,g,b))
    #temp = df[((euclidean_distance([df["R"],df["G"],df["B"]],color_vector))) & (df.index != index)]
    # temp = df[((np.abs(r - df['R']) <= eps) & (np.abs(g - df['G']) <= eps) & (np.abs(b - df['B']) <= eps)) & (df.index != index)]
    # print(temp)
    # temp = df[(manhattan_distance([(df['R'],df['G'],df['B'])],color_vector) <= eps) & (df.index != index)]
    # print(temp)
    list_temp = []
    for i,row in df.iterrows():
        if(i != index):
            _r,_g,_b = row["R"],row["G"],row["B"]
            this_color_vector = np.array((_r,_g,_b))
            #print(euclidean_distance(color_vector,this_color_vector))
            if(manhattan_distance(color_vector,this_color_vector) <= eps):
                list_temp.append(this_color_vector)
    temp = pd.DataFrame(list_temp,columns=["R","G","B"])
    #print(temp)

    # #temp = df[(euclidean_distance(np.array((df["R"],df["G"],df["B"])),color_vector))]
    # print(temp)
    if(len(temp) >= minimumPts):
        return (temp.index,True,False,False)
    elif(len(temp) < minimumPts and len(temp) > 0):
        return (temp.index,False,True,False)
    elif(len(temp) == 0):
        return (temp.index,False,False,True)

def predict2(eps,minimumPts,df):
    label_clusters = 1

    stack_c = set()
    visiteNo = list(df.index)
    clusters = []
    while(len(visiteNo) != 0):
        first_pt = True # On identifie que le point actuel est le premier point du cluster
        stack_c.add(random.choice(visiteNo)) # On choisi un point non visité
        while (len(stack_c) != 0): # Tant que la pile n'est pas vide c'est qu'il y a encore des points à traiter dans le cluster
            c_index = stack_c.pop() # Dernier point de la pile
            # Verifie si le point à l'index c_index est un coeur, une bordure ou bien noise
            indexVoisins, coeur,bordure,noise = type_point(eps,minimumPts,df,c_index)

            if(bordure and first_pt):
                clusters.append((c_index,0)) # Cluster label 0 pour les points isolé
                clusters.extend(list(zip(indexVoisins,[0 for _ in range(len(indexVoisins))])))

                visiteNo.remove(c_index)
                visiteNo = [e for e in visiteNo if e not in indexVoisins]
                continue
            visiteNo.remove(c_index)
            indexVoisins = set(indexVoisins) & set(visiteNo)
            if(coeur):
                first_pt =False
                clusters.append((c_index,label_clusters))
                stack_c.update(indexVoisins)
            elif(bordure):
                clusters.append((c_index,label_clusters))
                continue
            elif(noise):
                clusters.append((c_index,0))
                continue
        if not first_pt:
            label_clusters+=1

    return clusters


def predict3(eps, minimumPts, df):
    label_clusters = 1
    labels = np.full(shape=df.shape[0],fill_value=0)
    stack_c = set()
    visiteNo = list(df.index)
    clusters = []
    list_clusters = {}
    list_clusters[0] = []
    list_clusters[1] = []
    while (len(visiteNo) != 0):
        first_pt = True  # On identifie que le point actuel est le premier point du cluster
        stack_c.add(random.choice(visiteNo))  # On choisi un point non visité
        while (len(stack_c) != 0):  # Tant que la pile n'est pas vide c'est qu'il y a encore des points à traiter dans le cluster
            c_index = stack_c.pop()  # Dernier point de la pile
            # Verifie si le point à l'index c_index est un coeur, une bordure ou bien noise
            indexVoisins, coeur, bordure, noise = type_point(eps, minimumPts, df, c_index)

            if (bordure and first_pt):
                clusters.append((c_index, 0))  # Cluster label 0 pour les points isolé
                labels[c_index] = 0
                list_clusters[0].append(c_index)
                clusters.extend(list(zip(indexVoisins, [0 for _ in range(len(indexVoisins))])))

                visiteNo.remove(c_index)
                visiteNo = [e for e in visiteNo if e not in indexVoisins]
                continue
            visiteNo.remove(c_index)
            indexVoisins = set(indexVoisins) & set(visiteNo)
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
    print(clusters)
    print(labels)
    print(list_clusters)
    print(len(list_clusters))
    return labels,list_clusters
   # return clusters

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


def get_voisin(indexPixel):
    """
    Donne la liste d'index des voisins d'un certain pixel
    Il faut que la distance soit plus petite que la valeur d'epsilon pour que 2 pixels soient voisin
    """
    voisinsPixel = []
    indexes = np.arange(len(image))
    for index, _sample in enumerate(image[indexes!=indexPixel]):
        distance = euclidean_distance(image[indexPixel],_sample)
        if(distance < epsilon):
            voisinsPixel.append(index)
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

def launch():
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

    print(len(clusters))
    for i in range(len(clusters)):
        print(centeroidnp(image[clusters[i]]))

    # the cluster centroids is our color palette
    identified_palette = np.array(list_centroid).astype(int)
    print(identified_palette)
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

if __name__ == '__main__':
    start = t.time()
    global image
    image = img_load()
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    data = pd.DataFrame(image,columns=["R","G","B"])


    print(data)
    #launch()
    labels , list_clusters= predict3(15,3,data)
    for key in list_clusters.keys():
        centeroidnp(image[list_clusters[key]])
    # the cluster centroids is our color palette
    identified_palette = np.array(list_centroid).astype(int)
    # recolor the entire image
    recolored_img = np.copy(image)
    for index in range(len(recolored_img)):
        if(labels[index] == 0): # Il faut prendre en compte le dernier label qui correspond aux pixels ne faisant partie d'aucun cluster
            recolored_img[index] = np.array([255,255,255])
        else:
            recolored_img[index] = identified_palette[labels[index]]
    recolored_img = recolored_img.reshape(width, height, 3)
    plt.imshow(recolored_img)
    plt.show()
    end = t.time()
    print(str(end-start) + " secondes")

## Ne pas passer par un tableau
## Complexite algorithmique en n