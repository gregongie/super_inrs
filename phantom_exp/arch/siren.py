#taken from official SIREN repo: https://github.com/vsitzmann/siren
import torch
from torch import nn
import numpy as np

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    # def forward_with_intermediate(self, input): 
    #     # For visualization of activation distributions
    #     intermediate = self.omega_0 * self.linear(input)
    #     return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.depth = hidden_layers+1
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        # return output, coords        
        return output

    # def forward_with_activations(self, coords, retain_grad=False):
    #     '''Returns not only model output, but also intermediate activations.
    #     Only used for visualizing activations later!'''
    #     activations = OrderedDict()

    #     activation_count = 0
    #     x = coords.clone().detach().requires_grad_(True)
    #     activations['input'] = x
    #     for i, layer in enumerate(self.net):
    #         if isinstance(layer, SineLayer):
    #             x, intermed = layer.forward_with_intermediate(x)
                
    #             if retain_grad:
    #                 x.retain_grad()
    #                 intermed.retain_grad()
                    
    #             activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
    #             activation_count += 1
    #         else: 
    #             x = layer(x)
                
    #             if retain_grad:
    #                 x.retain_grad()
                    
    #         activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
    #         activation_count += 1

    #     return activations
    
    def weight_decay(self):
        wdloss=0
        for k in range(0, self.depth):
            wdloss += 0.5*(self.net[k].linear.weight**2).sum()
        wdloss += 0.5*(self.net[self.depth].weight**2).sum()
        return wdloss
    
    def register(self,vars): #no extra variables to register
        return 
  
    
#########################

def build(arch_options):
    in_features = 2
    depth = arch_options["depth"]
    hidden_features = arch_options["width"]
    hidden_layers = depth-1
    out_features = 1
    first_omega_0 = arch_options["first_omega_0"]
    hidden_omega_0 = arch_options["hidden_omega_0"]
    return Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)