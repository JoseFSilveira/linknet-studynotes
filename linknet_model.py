import torch
from torch import nn
from torchvision import models

class LinkNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Carrega o modelo ResNet-18 pré-treinado e extrai as camadas necessárias para o encoder
        # OBS: Pesos tambem podem ser chamado como .DEFAULT, mas a versao DEFAULT pode ser diferente dependendo da versao do torchvision
        resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Criando o bloco inicial do LinkNet, composto pelas primeiras camadas do ResNet-18, seguindo a arquitetura original do LinkNet
        self.initial_block = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool
        )

        # Criando os blocos do encoder a partir das camadas do ResNet-18, seguindo a arquitetura original do LinkNet
        self.encoder1 = resnet_model.layer1
        self.encoder2 = resnet_model.layer2
        self.encoder3 = resnet_model.layer3
        self.encoder4 = resnet_model.layer4

        # OBS: O LinkNet nao possui as duas camadas finais do ResNet-18 (avgpool e fc), visto que o LinkNet eh uma modelo de segmentacao e nao de classificacao.

        # Congelando o calculo do gradiente e atualizacao de parametros para todas as camadas do modelo
        # for param in self.parameters():
        #     param.requires_grad = False

    # def train(self, mode: bool=True):
    #     '''
    #     Essa funcao garante que o modelo LinkNetEncoder esteja sempre em modo de avaliacao, mesmo quando o modelo LinkNet estiver em modo de treinamento
    #     O encoder do LinkNet eh baseado no ResNet-18 pré-treinado e nao deve ser treinado junto com o decoder do LinkNet.
    #     '''
    #     super().train(mode=False)
    #     return self
    
    def forward(self, x):
        e1 = self.encoder1(self.initial_block(x))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        #e4 = self.encoder4(e3)

        return e1, e2, e3, self.encoder4(e3)
    

class LinkNetDecoder(nn.Module):

    class LinkNetDecoderBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            self.first_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True)
            )
            self.upsample_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels//4),
                nn.ReLU(inplace=True)
            )
            self.last_conv = nn.Sequential(
                nn.Conv2d(in_channels//4, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.last_conv(self.upsample_conv(self.first_conv(x)))
        
    def __init__(self, out_channels, dropout: bool=False):
        super().__init__()
        
        if dropout:
            self.decoder4 = nn.Sequential(
                nn.Dropout2d(p=0.2),
                self.LinkNetDecoderBlock(512, 256))
        else:
            self.decoder4 = self.LinkNetDecoderBlock(512, 256)
        
        self.decoder3 = self.LinkNetDecoderBlock(256, 128)
        self.decoder2 = self.LinkNetDecoderBlock(128, 64)
        self.decoder1 = self.LinkNetDecoderBlock(64, 64)

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Alterando a ultima camada do modelo LinkNet para nao aumentar a resolucao da imagem, visto que o modelo LinkNet original aumenta a resolucao da imagem em 2x
            nn.Conv2d(32, out_channels, kernel_size=2, stride=1, padding='same'),
            #nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=1, padding=0, output_padding=0),
            #nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, e1, e2, e3, e4):
        '''
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(e3)
        d2 = e1 + self.decoder2(e2)

        return self.final_block(self.decoder1(d2))
        '''
        return self.final_block(self.decoder1(e1 + self.decoder2(e2 + self.decoder3(e3 + self.decoder4(e4)))))
    

class LinkNet(nn.Module):
    def __init__(self, out_channels: int):
        '''
        Cria o modelo LinkNet, composto por um encoder baseado no ResNet-18 e um decoder personalizado, seguindo a arquitetura original do LinkNet.
        Eh esperado que o numero de canais de entrada seja 3, visto que o modelo foi projetado para trabalhar com imagens RGB.
        O numero de canais de saida deve ser igual ao numero de classes do problema de segmentacao.
        '''
        super().__init__()

        self.dummy_param = nn.Parameter(torch.empty(size=(0,))) # para tirar informacao de device do modelo

        self.encoder = LinkNetEncoder()
        self.decoder = LinkNetDecoder(out_channels, dropout=True)

    def forward(self, x):
        # * para desempacotar a tupla retornada pelo encoder, passando cada elemento como argumento separado para o decoder.
        return self.decoder(*self.encoder(x))