import torch
import torch.nn as nn
from models.CentralAttention import CentralAttention


from models.p2t import PyramidPoolingTransformer
from functools import partial
from models.GlobalLocalSE import GlobalLocalSE

cfg5 = {
"hidden_size" : 512,
"mlp_dim" : 512*4,
"num_heads" : 8,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg4 = {
"hidden_size" : 320,
"mlp_dim" : 320*4,
"num_heads" : 8,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg3 = {
"hidden_size" : 128,
"mlp_dim" : 128*4,
"num_heads" : 8,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BioSalNet(nn.Module):

    def __init__(self):
        super(BioSalNet, self).__init__()
        self.backbone = P2Tenconder()
        self.globallocalse5 = GlobalLocalSE(in_channels=512, spatial_size=11 * 11, cfg=cfg5)
        self.globallocalse4 = GlobalLocalSE(in_channels=320, spatial_size=22 * 22, cfg=cfg4)
        self.globallocalse3 = GlobalLocalSE(in_channels=128, spatial_size=44 * 44, cfg=cfg3)

        self.centralattention5 = CentralAttention(512, 11, 11, 16)
        self.centralattention4 = CentralAttention(320, 22, 22, 16)
        self.centralattention3 = CentralAttention(128, 44, 44, 8)

        self.decoder = Decoder()

    def forward(self, x, depth):
        x = self.backbone(x)
        depth = self.backbone(depth)
        xp2, xp3, xp4, xp5 = x
        xr2, xr3, xr4, xr5 = depth

        f5 = self.globallocalse5(xp5, xr5)
        f4 = self.globallocalse4(xp4, xr4)
        f3 = self.globallocalse3(xp3, xr3)

        f5 = self.centralattention5(f5)
        f4 = self.centralattention4(f4)
        f3 = self.centralattention3(f3)

        x = self.decoder(f5,f4,f3)
        return x


class P2Tenconder(nn.Module):
    def __init__(self):
        super(P2Tenconder, self).__init__()
        self.p2tmodel = PyramidPoolingTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 9, 3])
        self.p2tmodel.load_state_dict(torch.load(r'p2t_small.pth'))

    def forward(self, x):
        outputs = self.p2tmodel(x)
        return outputs


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.convbnrelu5 = ConvBNRelu(512, 320, 3, 1,1)
        self.convbnrelu4 = ConvBNRelu(320, 128, 3, 1,1)
        self.convbnrelu3 = ConvBNRelu(128, 64, 3, 1,1)
        self.convbnrelu2 = ConvBNRelu(64, 32, 3, 1,1)
        self.conv1 = nn.Conv2d(32, 1, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.sigmoid = nn.Sigmoid()

        self.sideout4 = nn.Sequential(
            ConvBNRelu(128, 64, 3, 1, 1),
            ConvBNRelu(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
        )
        self.sideout3 = nn.Sequential(
            ConvBNRelu(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
        )



    def forward(self, f5,f4,f3):

        f5 = self.convbnrelu5(f5)
        f5 = self.upsample(f5)

        f4 = f5 * f4
        f4 = self.convbnrelu4(f4)
        f4 = self.upsample(f4)

        f3 = f4 * f3
        f3 = self.convbnrelu3(f3)
        f3 = self.upsample(f3)

        f2 = self.convbnrelu2(f3)
        f2 = self.upsample4(f2)
        f = self.conv1(f2)
        f = self.sigmoid(f)
        f = f.squeeze(1)
        return f

