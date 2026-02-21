import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from pathlib import Path
# Mais informacoes em https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html#torchmetrics.classification.MulticlassJaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex

from train_model import NUM_CLASSES

def print_results(model_results: dict) -> None:

    if model_results is not None:
        
        # Gerar Figuras
        plt.figure(figsize=(20,10))
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

        epochs = range(1, len(model_results['train_loss'])+1)
        ax[0].scatter(epochs, model_results['train_loss'], label='Train Loss', c='blue')
        ax[0].scatter(epochs, model_results['val_loss'], label='Val Loss', c='orange')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].set_title('Loss Curves')

        IoU_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=NUM_CLASSES-1, average='none')
        IoU_metric.plot(model_results['val_IoU'], ax=ax[1])
        ax[1].set_title('IoU Curves')

        iIoU_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=NUM_CLASSES-1, average='weighted')
        iIoU_metric.plot(model_results['val_iIoU'], ax=ax[2])
        ax[2].set_title('iIoU Curve')

        plt.show()

    else:
        print("Resultados do modelo nao encontrados.")


# Imprimindo imagens e segmentações do dataset
def img_show(imgs: list[torch.Tensor], smnts1: list[torch.Tensor], smnts2: list[torch.Tensor]=None, n: int=5, col_names: list = None, cmap: mcolors.LinearSegmentedColormap = None) -> None:

    '''
    Essa funcao imprime as imagens e ssegmentacoes do dataset,
    caso smnts2 seja fornecido, entao as segmentacoes de smnts1 e smnts2 sao impressas lado a lado,
    caso contrario, apenas as segmentacoes de smnts1 sao impressas
    '''

    # Definindo Vazriavel auxiliar paraa definir se os titulos das colunas devem er mostrados ou nao, dependendo se col_names foi fornecido e se a quantidade de nomes informados estiver correta
    if col_names is not None:
        if smnts2 is not None and len(col_names) == 3:
            show_col_names = True
        elif smnts2 is None and len(col_names) == 2:
            show_col_names = True
        else:
            show_col_names = False
            print("Quantidade de nomes de colunas informados nao corresponde a quantidade de colunas a serem mostradas. Os titulos das colunas nao serao mostrados.")

    # Definindo o numero de colunas e o titulo de cada coluna, dependendo se smnts2 foi fornecido ou nao
    if smnts2 is not None:
        fig_dim = 15
        num_cols = 3
    else:
        fig_dim = 10
        num_cols = 2

    # Criando a figura e os eixos para imprimir as imagens e segmentacoes
    fig, axes = plt.subplots(nrows=n, ncols=num_cols, figsize=(fig_dim, fig_dim))

    # Colocando Titulo para cada Coluna caso col_names tenha sido fornecido corretamente
    if show_col_names:
        for ax, col in zip(axes[0], col_names):
            ax.set_title(col)

    for i in range(n):
        axes[i, 0].imshow(imgs[i].permute(1,2,0)) # permute para mudar a ordem dos canais e converter um tensor para imagem
        axes[i, 0].axis('off')
        axes[i, 1].imshow(smnts1[i], cmap=cmap)
        axes[i, 1].axis('off')
        if smnts2 is not None:
            axes[i, 2].imshow(smnts2[i], cmap=cmap)
            axes[i, 2].axis('off')
    plt.show()


def predict_mask(model: torch.nn.Module, img: torch.Tensor, device: torch.device='cpu') -> torch.Tensor:
    smnt_logits = model(img.unsqueeze(dim=0).to(device))
    smnt_mask = torch.softmax(smnt_logits, dim=1).argmax(dim=1).squeeze(dim=0).cpu()
    return smnt_mask


def test_model(model: torch.nn.Module, dataset, n:int = 5, device: torch.device='cpu', cmap: mcolors.LinearSegmentedColormap = None) -> None:
    img_list = []
    smnt_list = []

    # Aloca o modelo para o dispositivo correto, caso necessario
    if model.dummy_param.device != device:
        model.to(device)

    # Gera as listas de imagens e mascaras e chapa a funcao de imprimir
    for i in random.sample(range(len(dataset)), k=n):
        img = dataset[i][0]
        smnt_mask = predict_mask(model, img, device)
        img_list.append(img)
        smnt_list.append(smnt_mask)
    img_show(imgs=img_list, smnts1=smnt_list, n=n, col_names=['Imagem Original', 'Predicao'], cmap=cmap)


def dataset_show(dataset, n:int = 5, predict_masks: bool=False, model: torch.nn.Module=None, device: torch.device='cpu', cmap: mcolors.LinearSegmentedColormap = None) -> None:

    img_list = []
    smnt_list = []
    for i in random.sample(range(len(dataset)), k=n):
        img, smnt = dataset[i]
        img_list.append(img)
        smnt_list.append(smnt)

    # Realiza as predicoes do modelo, caso seja fornecido um
    if predict_masks and model is not None:

        # Aloca o modelo para o dispositivo correto, caso necessario
        if model.dummy_param.device != device:
            model.to(device)
        
        # Gerando as mascaras preditas pelo modelo para cada imagem do dataset
        pred_smnt_list = []
        for img in img_list:
            pred_smnt = predict_mask(model, img, device)
            pred_smnt_list.append(pred_smnt)

        col_names = ['Imagem Original', 'Ground Truth', 'Predicao']
    
    # Caso contrario apenas aloca None para a lista de predicoes, para que a funcao de imprimir saiba que nao deve imprimir as mascaras preditas
    else:
        pred_smnt_list = None
        col_names = ['Imagem Original', 'Ground Truth']

    img_show(imgs=img_list, smnts1=smnt_list, smnts2=pred_smnt_list,
             n=n,col_names=col_names, cmap=cmap)


def load_state_dict(model: torch.nn.Module, name: str, load_reults: bool=False, device: torch.device='cpu') -> torch.nn.Module | tuple[torch.nn.Module, dict]:
    
    # Carregando apenas os parametros (state_dict()), pois isso flexibiliza o modelo e evita erros de incompatibilidade com parametros e caminhos do modelo original
    # OBS: torch.load() carrega o modelo inteiro, nao apenas os parametros
    MODEL_PATH = Path.cwd() / Path("saved_models")
    MODEL_NAME = name + ".pth"
    SAVED_MODEL_PATH = MODEL_PATH / MODEL_NAME
    if SAVED_MODEL_PATH.is_file():
        model.load_state_dict(torch.load(f=SAVED_MODEL_PATH))
    else:
        print(f"Modelo {name} nao encontrado.")

    # Se load_results for True, entao o modelo e carregado e testado, e os resultados sao impressos
    if load_reults:
        REULTS_NAME = name + "_results.pt"
        RESULTS_SAVE_PATH = MODEL_PATH / REULTS_NAME
        if RESULTS_SAVE_PATH.is_file():
            with open(RESULTS_SAVE_PATH, "rb") as f:
                model_results = torch.load(f)
        else:
            model_results = None
            print(f"Resultados do modelo {name} nao encontrados.")
        return model, model_results
    
    else:
        return model