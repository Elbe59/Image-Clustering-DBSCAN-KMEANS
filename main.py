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
    if not os.path.exists('output/'):
        os.makedirs('output/')

    path = './output/RESULT-K='+str(nb_cluster)+'-'+img_name
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


if __name__ == '__main__':
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
    save_results(image,'le-roi-lion.jpg',nb_cluster)