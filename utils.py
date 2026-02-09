import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# Imprimindo imagens e segmentações do dataset
def img_show(imgs: list[torch.Tensor], smnts: list[torch.Tensor], n: int=5, cmap: mcolors.LinearSegmentedColormap = None) -> None:    
    plt.figure(figsize=(10,10))
    for i in range(n):
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(imgs[i].permute(1,2,0)) # permute para mudar a ordem dos canais e converter um tensor para imagem
        plt.axis('off')
        plt.subplot(n, 2, 2*i+2)
        plt.imshow(smnts[i], cmap=cmap)
        plt.axis('off')

def dataset_show(dataset, n:int = 5, cmap: mcolors.LinearSegmentedColormap = None) -> None:
    img_list = []
    smnt_list = []
    for i in random.sample(range(len(dataset)), k=n):
        img, smnt = dataset[i]
        img_list.append(img)
        smnt_list.append(smnt)
    img_show(img_list, smnt_list, n, cmap)