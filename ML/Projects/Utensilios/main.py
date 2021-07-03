import os
import torch
import torchvision
from utensilios import UtensiliosDetection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import random



device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

train_dataset = UtensiliosDetection("D:\\Repositorios\\SIS421\\ML\\Projects\\Utensilios", image_set='train')
print(len(train_dataset))
print(train_dataset[100])

# utensilios_classes = {'Cucharas':0, 'Tenedores':1}
utensilios_classes = ["Cucharas", "Tenedores"]
# voc_classes = ["background",
#             "aeroplane",
#             "bicycle"]

def get_sample(ix):
    img, label = train_dataset[ix]
    # print("+++"*10)
    # print(img)
    # print("..."*10)
    # print(label)
    # print("+++"*10)
    img_np = np.array(img)

    anns = label['annotation']['object']
    if type(anns) is not list:
        anns = [anns]
    # labels = np.array([utensilios_classes.index(ann['name']) for ann in anns])
    # labels = np.array([utensilios_classes[ann['folder']] for ann in anns])
    labels = list
    # print(str(label['annotation']['folder'][0]))
    labels = [utensilios_classes.index(label['annotation']['folder'][0])]
    # labels = [utensilios_classes.index('Cucharas')]
    bbs = [ann['bndbox'] for ann in anns]
    bbs = np.array([[int(bb['xmin']), int(bb['ymin']),int(bb['xmax'])-int(bb['xmin']),int(bb['ymax'])-int(bb['ymin'])] for bb in bbs])
    anns = (labels, bbs)
    return img_np, anns

def plot_anns(img, anns, ax=None, bg=-1, classes=utensilios_classes):
    # anns is a tuple with (labels, bbs)
    # bbs is an array of bounding boxes in format [x_min, y_min, width, height]
    # labels is an array containing the label
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    labels, bbs = anns
    for lab, bb in zip(labels, bbs):
        if bg == -1 or lab != bg:
            x, y, w, h = bb
            rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            text = ax.text(x, y - 10, classes[lab], {'color': 'red'})
            text.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            ax.add_patch(rect)

img_np, anns = get_sample(100)
print(img_np.shape)
print("--"*20)
print(anns)