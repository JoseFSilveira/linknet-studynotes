import torch
from torch import nn

class LinkNetEncoder(nn.Module):

    class LinkNetEncoderBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()

            self.downsample_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.normal_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.downsample_skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(out_channels)
            )
            self.normalize = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            x = self.downsample_skip(x) + self.normal_conv(self.downsample_conv(x))
            return self.normalize(x) + self.normal_conv(self.normal_conv(x))
    
    def __init__(self):
        super().__init__()

        self.encoder1 = self.LinkNetEncoderBlock(64, 64)
        self.encoder2 = self.LinkNetEncoderBlock(64, 128)
        self.encoder3 = self.LinkNetEncoderBlock(128, 256)
        self.encoder4 = self.LinkNetEncoderBlock(256, 512)
    
    def forward(self, x):
        e1 = self.encoder1(x)
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
        
    def __init__(self, dropout: bool=False):
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
        
    def forward(self, e1, e2, e3, e4):
        '''
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(e3)
        d2 = e1 + self.decoder2(e2)

        return self.decoder1(d2)
        '''
        return self.decoder1(e1 + self.decoder2(e2 + self.decoder3(e3 + self.decoder4(e4))))
    

class LinkNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.dummy_param = nn.Parameter(torch.empty(size=(0,))) # para tirar informacao de device do modelo

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.encoder = LinkNetEncoder()
        self.decoder = LinkNetDecoder(dropout=True)

        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # * para desempacotar a tupla retornada pelo encoder, passando cada elemento como argumento separado para o decoder.
        return self.final_block(self.decoder(*self.encoder(self.initial_block(x))))