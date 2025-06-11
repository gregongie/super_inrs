import torch
from torch import nn
import numpy as np

def get_freq_gridded(K0):
    size = 2*K0+1
    freq = torch.fft.fftfreq(size,d=1/size)
    freq_x,freq_y = torch.meshgrid(freq,freq,indexing="ij")
    #select gridded frequencies in a rectangular half space, skipping (0,0) frequency
    mask0 = (torch.max(torch.abs(freq_x),torch.abs(freq_y)) <= K0) & (freq_x >= 0) & ~((freq_y<=0) & (freq_x == 0))
    B = torch.stack((freq_x[mask0 ],freq_y[mask0]), dim=-1)
    B = B.view(-1,2)
    return B

class FFReLUDeep(nn.Module):
    def __init__(self,B,width,depth):
        super().__init__()
        # self.B = B #fourier features frequencies matrix, size [nfreq,2]
        self.register_buffer("B", B) #fourier features frequencies matrix, size [nfreq,2]
        self.width = width
        self.depth = depth
        # define ReLU MLP
        layers = [nn.Linear(2*B.shape[0]+1, width, bias=False), nn.ReLU()]
        for i in range(depth-2):
            layers += [nn.Linear(width,width), nn.ReLU()]
        layers += [nn.Linear(width,1)]
        self.MLP = nn.Sequential(*layers)
        
    def ff_mapping(self,x):
        #x is a vector of coordinates, size=[input_dim]
        #B is the frequencies matrix, size=[#freq,input_dim]
        x_proj = (2.0*np.pi*x) @ self.B.T
        return torch.cat([np.sqrt(2)*torch.sin(x_proj), np.sqrt(2)*torch.cos(x_proj), torch.ones_like(x_proj[:,0])[:,None]], axis=-1) #append constant feature to ff_mapping

    def forward(self,xin):
        return self.MLP(self.ff_mapping(xin))

    def weight_decay(self):
        wdloss = 0
        for k in range(0,2*self.depth,2):
            wdloss += 0.5*(self.MLP[k].weight**2).sum()
        return wdloss
    
    def register(self,vars): #no extra variables to register
        return 

def build(arch_options):
    if arch_options["ff_freq"] == "random":
        gen = torch.Generator()
        gen.manual_seed(arch_options["ff_seed"])
        B = arch_options["sigma"]*torch.randn((arch_options["num_freq"],2),generator=gen)
    else:
        B = get_freq_gridded(arch_options["K0"])

    width = arch_options["width"]
    depth = arch_options["depth"]

    return FFReLUDeep(B,width,depth)