import torch
from torch import nn
import numpy as np

# taken from: https://gist.github.com/Yibo-Wen/0f982098e8c65bdf3ff1442530cd97cc
# note: this is a "pure" pytorch implementation of Instant-NGP, which is slower
# than the compiled versions that use the tinycuda-nn library

'''
Multiresolution Hash Encoding from Nvidia Instant-NGP
Please read the Instant-NGP paper Section 3 to understand the code
https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
'''

'''
All parameters use the same name as in the paper
'''
class MultiHash(nn.Module):

    input_dim: int # dimension of input, currently only support 1 to 7
    output_dim: int # dimension of output, 3 for RGB, 1 for Grayscale
    T: int # size of hash table, which should be 2^T (14 to 24)
    L: int # number of levels (8,12,16)
    F: int # number of features (2,4,8)
    N_min: int # min level resolution (16)
    N_max: int # max level resolution (512 to 524288)
    b: float # step size between each level [1.26,2]

    _hash_tables: nn.Parameter # L hash tables with shape [2^T,F]
    _resolutions: nn.Parameter # resolutions of all levels with shape [1,1,L,1]
    _prime_numbers: nn.Parameter # list of preset big hashing prime numbers
    _voxel_border_adds: nn.Parameter # helper tensor of shape [1,input_dim,1,2^input_dim]
    MLP: nn.ModuleList # default MLP

    '''
    N_max should be half of the picture width for gigapixel image reconstruction
    '''
    def __init__(self, input_dim:int, output_dim:int, T:int=14, L:int=16, F:int=2, N_min:int=16, N_max:int=512):
        super().__init__()
        assert input_dim<=7, "Current hash encoding only supports up to 7 dimensions."
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = T
        self.L = L
        self.F = F
        self.N_min = N_min
        self.N_max = N_max

        '''calculate step size for geometric series (equation in paper)'''
        self.b = np.exp((np.log(self.N_max)-np.log(self.N_min))/(self.L-1))
        '''
        integer list of each level resolution: N1, N2, ... , N_max
        shape [1,1,L,1]
        '''
        self._resolutions = nn.Parameter(
            torch.from_numpy(
                np.array([
                    np.floor(self.N_min * (self.b**i)) for i in range(self.L)
                ], dtype=np.int64)).reshape(1,1,-1,1),False)

        '''
        init hash tables fpr all levels
        each hash table shape [2^T,F]
        '''
        self._hash_tables = nn.ModuleList([
            nn.Embedding(2**self.T,self.F) for _ in range(self.L)
        ])
        for i in range(self.L):
            nn.init.uniform_(self._hash_tables[i].weight,-1e-4,1e-4) # init uniform random weight

        '''from Nvidia's Tiny Cuda NN implementation'''
        self._prime_numbers = nn.Parameter(
            torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]), requires_grad=False)

        '''
        a helper tensor which generates the voxel coordinates with shape [1,input_dim,1,2^input_dim]
        2D example: [[[[0, 1, 0, 1]],
                      [[0, 0, 1, 1]]]]
        3D example: [[[[0, 1, 0, 1, 0, 1, 0, 1]],
                      [[0, 0, 1, 1, 0, 0, 1, 1]],
                      [[0, 0, 0, 0, 1, 1, 1, 1]]]]
        For n-D, the i-th input add the i-th "row" here and there are 2^n possible permutations
        '''
        border_adds = np.empty((self.input_dim, 2**self.input_dim), dtype=np.int64)
        for i in range(self.input_dim):
            pattern = np.array(
                ([0] * (2**i) + [1] * (2**i)) * (2**(self.input_dim-i-1)),
                dtype=np.int64)
            border_adds[i, :] = pattern
        self._voxel_border_adds = nn.Parameter(
            torch.from_numpy(border_adds).unsqueeze(0).unsqueeze(2), False) # helper tensor of shape [1,input_dim,1,2^input_dim]

        # default MLP
        self.MLP = nn.ModuleList(
            [nn.Linear(self.L*self.F, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)]
        )

    def forward(self, x:torch.Tensor):
        '''
        forward pass, takes a set of input vectors and encodes them

        Args:
            x: A tensor of the shape [batch_size, input_dim] of all input vectors.

        Returns:
            A tensor of the shape [batch_size, L*F]
            containing the encoded input vectors.
        '''

        # 1. Scale each input coordinate by each level's resolution
        '''
        elementwise multiplication of [batch_size,input_dim,1,1] and [1,1,L,1]
        '''
        scaled_coords = torch.mul(x.unsqueeze(-1).unsqueeze(-1), self._resolutions) # shape [batch_size,input_dim,L,1]
        # compute the floor of all coordinates
        if scaled_coords.dtype in [torch.float32, torch.float64]:
            grid_coords = torch.floor(scaled_coords).type(torch.int64) # shape [batch_size,input_dim,L,1]
        else:
            grid_coords = scaled_coords # shape [batch_size,input_dim,L,1]
        '''
        add all possible permutations to each vertex
        obtain all 2^input_dim neighbor vertices at this voxel for each level
        '''
        grid_coords = torch.add(grid_coords, self._voxel_border_adds) # shape [batch_size,input_dim,L,2^input_dim]

        # 2. Hash the grid coords
        hashed_indices = self._fast_hash(grid_coords) # hashed shape [batch_size, L, 2^input_dim]

        # 3. Look up the hashed indices
        looked_up = torch.stack([
            # use indexing for nn.Embedding (check pytorch doc)
            # shape [batch_size,2^n,F] before permute
            # shape [batch_size,F,2^n] after permute
            self._hash_tables[i](hashed_indices[:,i]).permute(0, 2, 1)
            for i in range(self.L)
        ],dim=2) # shape [batch_size,F,L,2^n]

        # 4. Interpolate features using multilinear interpolation
        # 2D example: for point (x,y) in unit square (0,0)->(1,1)
        # bilinear interpolation is (1-x)(1-y)*(0,0) + (1-x)y*(0,1) + x(1-y)*(1,0) + xy*(1,1)
        weights = 1.0 - torch.abs(
            torch.sub(scaled_coords, grid_coords.type(scaled_coords.dtype))) # shape [batch_size,input_dim,L,2^input_dim]
        weights = torch.prod(weights, axis=1, keepdim=True) # shape [batch_size,1,L,2^input_dim]

        # sum the weighted 2^n vertices to shape [batch_size,F,L]
        # swap axes to shape [batch_size,L,F]
        # final shape [batch_size,L*F]
        output = torch.sum(torch.mul(weights, looked_up),
                         axis=-1).swapaxes(1, 2).reshape(x.shape[0], -1)

        # pass into MLP
        for layer in self.MLP:
            output = layer(output)
        return output

    def _fast_hash(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Implements the hash function proposed by NVIDIA.
        Args:
            x: A tensor of the shape [batch_size, input_dim, L, 2^input_dim].
               This tensor should contain the vertices of the hyper cuber
               for each level.
        Returns:
            A tensor of the shape [batch_size, L, 2^input_dim] containing the
            indices into the hash table for all vertices.
        '''
        tmp = torch.zeros((x.shape[0], self.L, 2**self.input_dim), # shape [batch_size,L,2^input_dim]
                          dtype=torch.int64,
                          device=x.device)
        for i in range(self.input_dim):
            tmp = torch.bitwise_xor(x[:, i, :, :] * self._prime_numbers[i], # shape [batch_size,L,2^input_dim]
                                    tmp)
        return torch.remainder(tmp, 2**self.T) # mod 2^T

    def weight_decay(self):
        depth = int((len(self.MLP)+1)/2)
        wdloss = 0.0
        for k in range(0,2*depth,2):
            wdloss += 0.5*(self.MLP[k].weight**2).sum()
        return wdloss
    
    def register(self,vars): #no extra variables to register
        return 


def build(arch_options):
    T = arch_options["T"]
    L = arch_options["L"]
    F = arch_options["F"]
    N_min = arch_options["N_min"]
    N_max = arch_options["N_max"]
    return MultiHash(2, 1, T, L, F, N_min, N_max)