import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision import tv_tensors  # <--- Importação necessária pára Augmentations
import matplotlib.colors as mcolors


class CityscapesLables:
    def __init__(self):

        # Criando listas dos nomes e das cores das classes treinaveis
        id_names = {}
        color_list = []
        lable_conversion = {}
        for c in datasets.Cityscapes.classes:

            # Adicionando valores ao dicionario de conversao de ids
            lable_conversion[c.id] = c.train_id if c.train_id != 255 else 19 # Mapeia a classe 'ignore' (train_id 255) para 19, que é o índice da última classe treinavel

            # Adicionando valores as listas de nomes e cores
            if c.train_id != -1 and c.train_id != 255:
                id_names[c.train_id] = c.name
                color_list.append(c.color)

        # Variavel para dicionario de nomes
        id_names.update({19: 'ignore'}) # Adiciona a classe 'ignore' com train_id -1
        self.id_names = id_names

        # Variavel para lista de cores
        train_colors_list = color_list
        train_colors_list.append((0,0,0)) # Adiciona a cor preta para a classe 'ignore'
        self.train_colors_list = train_colors_list

        # Criando o mapeamento para labels de treino
        lable_conversion.pop(-1) # Remove lablel 'ignore' que tem train_id -1
        self.lable_conversion = lable_conversion

    def get_cmap(self, name='CityscapesTrainColorMap'):
        # Criando o color map para imprimir imagens segmentadas
        train_color_map = ['#%02x%02x%02x' % color for color in self.train_colors_list]
        return mcolors.LinearSegmentedColormap.from_list(name, train_color_map, N=len(train_color_map))


# Criar Classe para dateset modificado
class AugmentedCityscapes(datasets.Cityscapes):
    def __init__(self, *args, data_augmentation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_augmentation = data_augmentation

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)

        # Aplicar data augmentation se estiver definido
        if self.data_augmentation:

            # tv_tensors informama ao v2 que o 'image' é uma Imagem e 'target' é uma Mascara.
            # Isso garante que rotacoes/flips sejam aplicados em ambos,
            # e que ajustes de cor (se houver) sejam aplicados APENAS na imagem.
            image, target = self.data_augmentation(tv_tensors.Image(image), tv_tensors.Mask(target))

        if self.split == 'train':
            image = v2.ToDtype(torch.float32, scale=True)(image) # normaliza para [0,1]

        return image, target