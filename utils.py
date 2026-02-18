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


def test_model(model: torch.nn.Module, dataset, n:int = 5, device: torch.device='cpu', cmap: mcolors.LinearSegmentedColormap = None) -> None:
        img_list = []
        smnt_list = []

        # Aloca o modelo para o dispositivo correto, caso necessario
        if model.dummy_param.device != device:
            model.to(device)

        # Gera as listas de imagens e mascaras e chapa a funcao de imprimir
        for i in random.sample(range(len(dataset)), k=n):
            img = dataset[i][0]
            smnt_logits = model(img.unsqueeze(dim=0).to(device))
            smnt_mask = torch.softmax(smnt_logits, dim=1).argmax(dim=1).squeeze(dim=0).cpu()
            img_list.append(img)
            smnt_list.append(smnt_mask)
        img_show(img_list, smnt_list, n, cmap)