import torch
from torch import nn
from torch import optim
from torch import optim
# Mais informacoes em https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html#torchmetrics.classification.MulticlassJaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex # Jaccard Index eh a mesma coisa que IoU (Intersection over Union) e esse opjeto pode calcular tambem iIoU
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

NUM_CLASSES = 20 # Numero de classes do dataset, incluindo a classe de ignorar (void)

class TestLinkNet:
  
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

        # Aloca o modelo para o dispositivo correto, caso necessario
        if self.model.dummy_param.device != self.device:
            self.model.to(self.device)

        # Definir loss fuinction e otimizador
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=NUM_CLASSES-1).to(device)
        self.optim_fn = optim.Adam(model.parameters(), lr=1e-4)

        # Definir metricas
        self.IoU_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=NUM_CLASSES-1, average='none').to(self.device)
        self.iIoU_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, ignore_index=NUM_CLASSES-1, average='weighted').to(self.device)

        # Criando dicionario para armazenar os resultados
        self.results = {"test_loss": 0,
                        "test_IoU": [],
                        "test_iIoU": 0}

        # Definindo pasta onde modelos estao salvos
        self.MODEL_PATH = Path.cwd() / Path("saved_models") # Armazenar modelos salvos numa pasta nova dentro da pasta do projeto


    # -------------------------------- #
    # -- MODEL TEST (CALL) FUNCTION -- #
    # -------------------------------- #
    def __call__(self, dataloader: torch.utils.data.DataLoader) -> None:

        self.model.eval()

        print("Realizando Teste do Modelo:")

        test_loss, test_iIoU = 0, 0
        test_IoU = torch.zeros(NUM_CLASSES) # Inicializa o IoU como um array de zeros para cada classe

        with torch.inference_mode():
            for batch, (X, y) in enumerate(tqdm(dataloader)):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                test_loss += self.loss_fn(y_pred, y.long()).item() # Accumulate the scalar value

                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
                test_IoU += self.IoU_metric(y_pred_class, y).cpu() # Acumular IoU para cada classe
                test_iIoU += self.iIoU_metric(y_pred_class, y).cpu() # Acumular iIoU para cada classe
        
        self.results['test_loss'] = test_loss / len(dataloader)
        self.results['test_IoU'] = test_IoU / len(dataloader)
        self.results['test_iIoU'] = torch.clone(test_iIoU) / len(dataloader)

        # Imprimindo os resultados
        print(f"test_loss: {self.results['test_loss']:.4f} | " # Formatted to 4 decimal places for readability
                #f"test_IoU: {self.results['test_IoU']:.4f} | "
                f"test_iIoU: {self.results['test_iIoU']:.4f}")