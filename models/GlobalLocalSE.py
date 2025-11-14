import torch
import torch.nn as nn
from models.Transformer import Encoder, Block
from einops import rearrange
import torch.nn.functional as F


class GlobalLocalSE(nn.Module):
    def __init__(self, in_channels, spatial_size, cfg, reduction=16):
        super(GlobalLocalSE, self).__init__()
        self.localmodel = Local(in_channels)
        self.semodel = SEModel(in_channels, reduction)
        self.Global = Global(in_channels, spatial_size, cfg)

    def forward(self, x, x_depth):
        x_l = self.localmodel(x)
        x_se = self.semodel(x_depth)
        x = self.Global(x_l, x_se)
        return x





class Global(nn.Module):

    def __init__(self, in_channels, spatial_size, cfg):
        super(Global, self).__init__()

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=cfg['hidden_size'],
                                          kernel_size=1,
                                          stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, spatial_size, cfg['hidden_size']))

        self.attn = Block(cfg)

    def forward(self, x_g, x_l):
        x_g_a, x_g_b = x_g.shape[2], x_g.shape[3]

        # x = self.patch_embeddings(x)
        x_g = x_g.flatten(2)
        x_g = x_g.transpose(-1, -2)

        x_l_a, x_l_b = x_l.shape[2], x_l.shape[3]
        # x = self.patch_embeddings(x)
        x_l = x_l.flatten(2)
        x_l = x_l.transpose(-1, -2)

        x_g = x_g + self.position_embeddings
        x_l = x_l + self.position_embeddings

        x_g = self.attn(x_g, x_l)
        #         x_l = self.attn(x_l, x_g)

        x_g_B, x_g_n_patch, x_g_hidden = x_g.shape
        x_g = x_g.permute(0, 2, 1)
        x_g = x_g.contiguous().view(x_g_B, x_g_hidden, x_g_a, x_g_b)

        x_l_B, x_l_n_patch, x_l_hidden = x_l.shape
        x_l = x_l.permute(0, 2, 1)
        x_l = x_l.contiguous().view(x_l_B, x_g_hidden, x_l_a, x_l_b)

        #         x = x_g * x_l
        return x_g






class Local(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(Local, self).__init__()
        self.conv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2,
                                 stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                 stride=1)
        self.relu = nn.ReLU(True)
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 4, 1),  # 提前定义而非在forward中动态创建
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h1 = self.conv5_5(x)
        h1 = self.relu(h1)
        h1 = self.bn(h1)

        h2 = self.conv3_3(x)
        h2 = self.relu(h2)
        h2 = self.bn(h2)

        h3 = self.conv1_1(x)
        h3 = self.relu(h3)
        h3 = self.bn(h3)

        weights = self.weight_generator(x)

        output = weights[:, 0:1] * h1 + weights[:, 1:2] * h2 + weights[:, 2:3] * h3 + weights[:, 3:4] * x
        return output


class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



