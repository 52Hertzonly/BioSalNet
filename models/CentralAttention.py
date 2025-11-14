import torch
import torch.nn as nn
from torch.nn import Softmax

soft = Softmax(dim=-1)


def center(w, n, device):
    array = [0] * n
    mid = n // 2
    for i in range(n):
        array[i] = i + 1 if i < mid else n - i

    two_d_array = [array[:] for _ in range(w)]  


    two_d_array = torch.tensor(two_d_array, device=device, dtype=torch.float32)
    two_d_array = two_d_array.unsqueeze(1)
    return soft(two_d_array)


class CentralAttention(nn.Module):
    def __init__(self, channels, h, w,  reduction=4):
        super(CentralAttention, self).__init__()

        self.h = h
        self.w = w


        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1)) 
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w)) 
        self.conv_1x1 = nn.Conv2d(in_channels=channels, out_channels=channels // reduction, kernel_size=1, stride=1,
                                  bias=False) 

        self.conv_1x1_f = nn.Conv2d(in_channels=(channels+channels//reduction), out_channels=channels, kernel_size=1, stride=1,
                                  bias=False)  

        self.relu = nn.ReLU()  
        self.bn = nn.BatchNorm2d(channels)  

        self.F_h = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1,
                             bias=False) 
        self.F_w = nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, stride=1,
                             bias=False)  

        self.sigmoid_h = nn.Sigmoid() 
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x)
        x_w = self.avg_pool_y(x)

        B,C,H,W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        tensor_device = x.device
        center_h = center(C, H, tensor_device)
        center_h = center_h.repeat(B, 1, 1, 1)
        center_h = center_h.permute(0, 1, 3, 2)

        center_w = center(C, W, tensor_device)
        center_w = center_w.repeat(B, 1, 1, 1)


        s_h = x_h * center_h
        s_w = x_w * center_w

        s_h = self.relu(s_h)
        s_w = self.relu(s_w)

        out = x * s_h * s_w

        out = self.relu(self.conv_1x1(out))
        out = torch.cat((x, out), 1)
        out = self.relu(self.bn(self.conv_1x1_f(out)))

        return out

