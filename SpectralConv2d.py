import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


class SpectralConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,models1,models2):
        super(SpectralConv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = models1
        self.modes2 = models2

        scale = 1 / (in_channels*out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels,out_channels,self.modes1,self.modes2,2))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels,out_channels,self.modes1,self.modes2,2))

    def compl_mul2d(self,input,weights):
        return torch.einsum("bixy,ioxy->boxy",input,torch.view_as_complex(weights))

    def forward(self,x):
        batch_size = x.shape[0]

        # Compute Fourier coeffients
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device) # why in this shape because becuase x_ft shape is batch_size , channels , heights , width//2+1
        

        out_ft[:,:,:self.modes1,:self.modes2] =\
                    self.compl_mul2d(x_ft[:,:,:self.modes1,:self.modes2],self.weights1) # low frequncey modes
        out_ft[:,:,-self.modes1:,:self.modes2] =\
                    self.compl_mul2d(x_ft[:,:,-self.modes1:,:self.modes2],self.weights2)  # high frequncey modes (in fourth dimenssion we only do till self.modes2 only those elements exist)

        x = torch.fft.irfft2(out_ft,s=(x.size(-2),x.size(-1))) # returns image (input is image and output is also image because of in between operations the output image will be different)
        return x
