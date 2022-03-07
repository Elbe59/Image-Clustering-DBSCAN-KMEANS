import os
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter.messagebox import showinfo
from threading import *
from PIL import ImageTk, Image

import K_means_final
import dbscan_final


root = Tk()
root.title('Interface TP2')

original_img = ""
img_result = ""
image_ref = ""
filepath = ""
box_configuration = Frame(root)
box_configuration.grid(row=1, column=1)
box_algo_dist = Frame(box_configuration)
box_algo_dist.grid(row=0, column=0)
box_param = Frame(box_configuration)
box_param.grid(row=0, column=2)
r_image = Frame()
r_image.grid(row=0, column=1)
type_algo = ""
type_dist = ""
width = ""
height = ""
new_height = ""
new_width = ""

def threading():
    # Call work function
    t1=Thread(target=work)
    t1.start()

def work():
    new_i, path,time = "", "",""
    if (type_algo == "DBSCAN"):
        new_i, path,time = dbscan_final.launch(eps=float(e1.get()), minPts=int(e2.get()), method=type_dist,
                                          image_path=filepath, new_width=int(new_width.get()),
                                          new_height=int(new_height.get()))
    if (type_algo == "KMEANS"):
        new_i, path,time = K_means_final.launch(k=int(e1.get()), nb_iterations=int(e2.get()), method=type_dist,
                                           image_path=filepath, new_width=int(new_width.get()),
                                           new_height=int(new_height.get()))
    basename = os.path.basename(path)
    add_results_image(adress=path, image_name=basename, image=new_i, temps=time)
    img_r = Label(r_image, image=img_result)
    img_r.grid(row=1, column=0, pady=5)

def add_original_image(adress):
    """
    Description :
    Ajouter à l'instance de tkinter l'image choisie.
    """
    global image_ref, width, height
    image_ref = Image.open(adress)
    width, height = image_ref.size

    new_w = width
    new_h = height
    if (new_w > 500):
        new_w = 500
    if (new_h > 350):
        new_h = 350
    IMG_SIZE = (new_w, new_h)
    image_ref = image_ref.resize(IMG_SIZE, Image.ANTIALIAS)
    image_ref = ImageTk.PhotoImage(image_ref)


def add_results_image(adress: str, image_name, image,temps):
    """
    Description :
    Ajoute l'image obtenu après clustering
    """
    global img_result
    image = Image.open(adress)
    width, height = image.size
    new_w = width
    new_h = height
    if (new_w > 500):
        new_w = 500
    if (new_h > 350):
        new_h = 350
    IMG_SIZE = (new_w, new_h)

    image = image.resize(IMG_SIZE, Image.ANTIALIAS)
    img_result = ImageTk.PhotoImage(image)
    basename = os.path.basename(adress)

    Label(r_image,text="Temps d'éxécution total = "+str(temps) + " secondes").grid(row=2,column=0)
    Label(r_image,text=adress).grid(row=0,column=0)


def on_closing():
    """
    Description :
    Stop le programme si la fenêtre est fermée.
    """
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.quit()

def select_file():
    """
    Méthode pour sélectionner un fichier depuis le répertoire et ensuite afficher les premières informations de l'image
    """
    global filepath, width, height
    global original_img
    global new_width, new_height
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    filetypes = (
        ('All files', '*.*'),
        ('image png', '.png'),
        ('image jpg', '.jpg'),
        ('text files', '*.txt'))

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir=ROOT_DIR,
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
    )
    o_image = Frame()
    for widgets in o_image.winfo_children(): # Permet de vider le contenu de la frame
        widgets.destroy()
    o_image.grid(row=0, column=0)
    filepath = filename
    add_original_image(filename)
    basename = os.path.basename(filepath)
    original_img = Label(o_image, image=image_ref)
    original_img.grid(row=1, column=0, pady=5)
    label_original = Label(o_image, text=basename, font=("Arial", 12))
    label_Taille = Label(o_image, text="Taille de l'image: " + str(width) + " x " + str(height), font=("Arial", 12))
    label_original.grid(row=0, column=0, pady=5)
    label_Taille.grid(row=2)
    label_fast = Label(o_image, text="Souhaitez vous rétrécir l'image pour plus de rapidité ?", font=("Arial", 12))
    label_fast.grid(row=3)
    frame_changeWH = Frame(o_image)
    for widgets in frame_changeWH.winfo_children(): # Permet de vider le contenu de la frame
        widgets.destroy()
    frame_changeWH.grid(row=4)
    l1 = Label(frame_changeWH)
    l2 = Label(frame_changeWH)
    conf1 = StringVar()
    conf2 = StringVar()
    new_width = Entry(frame_changeWH,textvariable=conf1)
    new_height = Entry(frame_changeWH,textvariable=conf2)

    l1["text"] = "Width"
    l2["text"] = "Height"
    conf1.set(width)
    conf2.set(height)

    l1.grid(row=4)
    l2.grid(row=5)
    new_width.grid(row=4, column=1)
    new_height.grid(row=5, column=1)
    choose()


def choose():
    Label(box_algo_dist, text="Choisissez l'algorithme et la distance à utiliser").grid(row=0, columnspan=2)
    type_algorithme = [
        ("KMEANS", 1),
        ("DBSCAN", 2),
    ]

    type_distance = [
        ("Euclidean distance", 1),
        ("Manhattan distance", 2),
    ]
    varAlgo = IntVar()
    varDist = IntVar()
    for txt, val in type_algorithme:
        Radiobutton(box_algo_dist, text=txt, variable=varAlgo, value=val, tristatevalue=0,
                    command=lambda t=txt: changeVarAlgo(t)).grid(column=0, row=val + 1)
    for txt, val in type_distance:
        Radiobutton(box_algo_dist, text=txt, variable=varDist, value=val, tristatevalue=0,
                    command=lambda t=txt: changeVarDist(t)).grid(column=1, row=val + 1)
    changeVarDist("Euclidean")
    changeVarAlgo("KMEANS")
    submit = Button(
        box_configuration,
        text='Lancer l\'exécution',
        command=lambda: launch_process(type_algo=type_algo, type_dist=type_dist, entry1=e1, entry2=e2)
    )
    submit.grid(column=1, row=1)


def add_processing_msg():
    txt_r = Label(r_image, text="Processing ...")
    txt_r.grid(row=0, column=0)


def launch_process(type_algo, type_dist, entry1, entry2):
    for widgets in r_image.winfo_children():  # Permet de vider le contenu de la frame
        widgets.destroy()
    threading()
    add_processing_msg()

def changeVarDist(txt):
    global type_dist
    if ("Manhattan" in txt):
        type_dist = "manhattan"
    if ("Euclidean" in txt):
        type_dist = "euclidean"


def changeVarAlgo(txt):
    """
    Prend en paramètre le type d'algorithme a utiliser et change l'affichage des paramètres que l'on peut modifier
    """
    global box_param
    global e1
    global e2
    global type_algo
    for widgets in box_param.winfo_children(): # Permet de vider le contenu de la frame
        widgets.destroy()
    Label(box_param, text="Paramètres à utiliser").grid(row=0, columnspan=2)
    l1 = Label(box_param)
    l2 = Label(box_param)
    e1 = Entry(box_param)
    e2 = Entry(box_param)
    conf1 = StringVar()
    conf2 = StringVar()
    if (txt == "DBSCAN"):
        l1["text"] = "Minimum Points"
        l2["text"] = "Epsilon"
        conf1.set("3")
        conf2.set("2")
        e1["textvariable"] = conf1
        e2["textvariable"] = conf2
        l1.grid(row=1)
        l2.grid(row=2)
        e1.grid(row=1, column=1)
        e2.grid(row=2, column=1)
        type_algo = "DBSCAN"
    if (txt == "KMEANS"):
        l1["text"] = "K"
        l2["text"] = "Nb Itérations"
        conf1.set("16")
        conf2.set("10")
        e1["textvariable"] = conf1
        e2["textvariable"] = conf2
        l1.grid(row=1)
        l2.grid(row=2)
        e1.grid(row=1, column=1)
        e2.grid(row=2, column=1)
        type_algo = "KMEANS"


def start_interface():
    """
    Description :
    Méhode principale qui initialise la fenêtre tkinter avec uniquement la possibilité de choisir un fichier à traiter
    """
    open_button = Button(
        root,
        text='Ouvrir un fichier',
        command=select_file
    )
    open_button.grid()

if __name__ == '__main__':
    start_interface()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()