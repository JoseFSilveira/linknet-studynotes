# tqdm.auto
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

class TrainLinkNet:
  
    def __init__(self, model: torch.nn.Module, device: torch.device, epochs: int = 5) -> None:
        self.model = model
        self.device = device
        self.epochs = epochs

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
        self.results = {"train_loss": [],
                        "train_IoU": [],
                        "train_iIoU": [],
                        "val_loss": [],
                        "val_IoU": [],
                        "val_iIoU": []}
        
        # Criando Pasta para salver Modelo
        self.MODEL_PATH = Path.cwd() / Path("saved_models") # Armazenar modelos salvos numa pasta nova dentro da pasta do projeto
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
    

    # ------------------------- #
    # -- SAVE MODEL FUNCTION -- #
    # ------------------------- #
    def save_model(self, name: str ="best_model.pth") -> None:

        # Criando Caminho para Salvar Modelo
        MODEL_NAME = name
        MODEL_SAVE_PATH = self.MODEL_PATH / MODEL_NAME

        # Salvando o modelo
        torch.save(obj=self.model.state_dict(), f=MODEL_SAVE_PATH) # Salva apenas os parametros do modelo (state_dict)


    # ------------------------- #
    # -- TRAIN STEP FUNCTION -- #
    # ------------------------- #
    def train_step(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        self.model.train()
        train_loss, train_iIoU = 0, 0
        train_IoU = torch.zeros(NUM_CLASSES) # Inicializa o IoU como um array de zeros para cada classe

        for batch, (X, y) in enumerate(tqdm(dataloader)):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            batch_loss = self.loss_fn(y_pred, y.long()) # Calculate loss for the current batch
            train_loss += batch_loss.item() # Accumulate the scalar value for reporting

            self.optim_fn.zero_grad()
            batch_loss.backward() # Perform backward pass on the current batch's loss
            self.optim_fn.step()

            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
            train_IoU += self.IoU_metric(y_pred_class, y).cpu() # Acumular IoU para cada classe
            train_iIoU += self.iIoU_metric(y_pred_class, y).cpu() # Acumular iIoU para cada classe

        train_loss /= len(dataloader)
        train_IoU /= len(dataloader)
        train_iIoU = torch.clone(train_iIoU) / len(dataloader)

        return train_loss, train_IoU, train_iIoU


    # ------------------------------ #
    # -- VALIDATION STEP FUNCTION -- #
    # ------------------------------ #
    def val_step(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        self.model.eval()
        val_loss, val_iIoU = 0, 0
        val_IoU = torch.zeros(NUM_CLASSES) # Inicializa o IoU como um array de zeros para cada classe

        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                val_loss += self.loss_fn(y_pred, y.long()).item() # Accumulate the scalar value

                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1).squeeze(dim=1) # Converte as probabilidades em classes preditas e remove a dimensão de canal extra
                val_IoU += self.IoU_metric(y_pred_class, y).cpu() # Acumular IoU para cada classe
                val_iIoU += self.iIoU_metric(y_pred_class, y).cpu() # Acumular iIoU para cada classe
        val_loss /= len(dataloader)
        val_IoU /= len(dataloader)
        val_iIoU = torch.clone(val_iIoU) / len(dataloader)

        return val_loss, val_IoU, val_iIoU
    

    # --------------------------------- #
    # -- MODEL TRAIN (CALL) FUNCTION -- #
    # --------------------------------- #
    def __call__(self, train_dataloader: torch.utils.data.DataLoader,
                    val_dataloader: torch.utils.data.DataLoader) -> None:

        # Cria variaveis para salvar melhor iIoU e em qual epoch foi atingido
        best_iIoU, best_epoch = 0, 0

        for epoch in range(self.epochs):
            print(f"EPOCH {epoch+1}/{self.epochs}")

            train_loss, train_IoU, train_iIoU = self.train_step(train_dataloader)
            val_loss, val_IoU, val_iIoU = self.val_step(val_dataloader)
            # Imprimindo o que esta acontecendo
            print(f"train_loss: {train_loss:.4f} | " # Formatted to 4 decimal places for readability
                    #f"train_IoU: {train_IoU:.4f} | "
                    f"train_iIoU: {train_iIoU:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    #f"val_IoU: {val_IoU:.4f} | "
                    f"val_iIoU: {val_iIoU:.4f}\n")

            # Armazena o modelo caso a iIoU for a maior registrada
            if val_iIoU > best_iIoU:
                best_iIoU = val_iIoU
                best_epoch = epoch
                self.save_model()

            # Passando os dados a cpu e os convertendo para float, caso necessario
            self.results['train_loss'].append(train_loss)
            self.results['train_IoU'].append(train_IoU)
            self.results['train_iIoU'].append(train_iIoU)
            self.results['val_loss'].append(val_loss)
            self.results['val_IoU'].append(val_IoU)
            self.results['val_iIoU'].append(val_iIoU)

        # Sinaliza fim do Treino
        print("Treino do modelo foi finalizado!\n"
              f"O modelo com melhor iIoU foi registrado no Epoch {best_epoch}.\n"
              f"Esse modelo foi salvo no caminho {self.MODEL_PATH}")