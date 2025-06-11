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

class FFReLUShallow(nn.Module):
    def __init__(self,B,width,reg_type):
        super().__init__()
        self.register_buffer("B", B) #fourier features frequencies matrix, size [nfreq,2]
        self.width = width
        self.reg_type = reg_type #"std", "mod1", or "mod2"
        self.inner = nn.Linear(2*B.shape[0]+1, width, bias=False)
        self.outer = nn.Linear(width, 1, bias=False)

    def register(self,vars): #register extra variables if needed
        self.res = vars["res"]  #rasterization resolution -- needed for reg_type="mod1","mod2"
        self.register_buffer("coords", vars["coords"]) #INR evaluation coordinates -- needed for reg_type="mod1","mod2"
        self.register_buffer("mask", vars["mask"]) #Fourier sampling mask -- only needed for reg_type="mod1"

    def ff_mapping(self,x):
        #x is a vector of coordinates, size=[input_dim]
        #B is the frequencies matrix, size=[#freq,input_dim]
        x_proj = (2.0*np.pi*x) @ self.B.T
        return torch.cat([np.sqrt(2)*torch.sin(x_proj), np.sqrt(2)*torch.cos(x_proj), torch.ones_like(x_proj[:,0])[:,None]], axis=-1)

    def hidden(self, xin):
        gammax = self.ff_mapping(xin)
        return torch.nn.functional.relu(self.inner(gammax)) #[W*gamma(x)]_+, dims [num_coords,width]

    def forward(self,xin):
        return self.outer(self.hidden(xin)) #a^T[W*gamma(x)]_+

    def weight_decay(self):
        a = self.outer.weight

        if self.reg_type == "std":
            W = self.inner.weight
            return 0.5*((W**2).sum() + (a**2).sum())
        
        if self.reg_type == "mod1":
            hidden = self.hidden(self.coords).T.view([self.width,self.res[0],self.res[1]]) #hidden layer output imgs [width,nx,ny]
            Ahidden = torch.fft.fft2(hidden,norm="ortho")[:,self.mask] #apply A to every hidden layer img [width,len(y)]
            return 0.5*((torch.real(Ahidden)**2 + torch.imag(Ahidden)**2).sum()/(self.res[0]*self.res[1])  + (a**2).sum())

        if self.reg_type == "mod2":
            hidden = self.hidden(self.coords).T
            return 0.5*((torch.sum(hidden,dim=1)**2).sum()/((self.res[0]*self.res[1])**2) + (a**2).sum())
        
    def unit_sizes(self):
        with torch.no_grad():
            a = self.outer.weight

            if self.reg_type == "std":
                W = self.inner.weight
                return torch.norm(W,dim=1)*torch.abs(a)
            
            if self.reg_type == "mod1":
                hidden = self.hidden(self.coords).T.view([self.width,self.res[0],self.res[1]]) #hidden layer output imgs [width,nx,ny]
                Ahidden = torch.fft.fft2(hidden,norm="ortho")[:,self.mask] #apply A to every hidden layer img [width,len(y)]
                return torch.norm(Ahidden,dim=1)*torch.abs(a)/np.sqrt(self.res[0]*self.res[1])

            if self.reg_type == "mod2":
                hout = self.hidden(self.coords).T
                return (torch.sum(hout,dim=1)*torch.abs(a))/(self.res[0]*self.res[1])

def build(arch_options):
    if arch_options["ff_freq"] == "random":
        gen = torch.Generator()
        gen.manual_seed(arch_options["ff_seed"])
        B = arch_options["sigma"]*torch.randn((arch_options["num_freq"],2),generator=gen)
    else:
        B = get_freq_gridded(arch_options["K0"])

    width = arch_options["width"]
    reg_type = arch_options["reg_type"]
    
    return FFReLUShallow(B, width, reg_type)
