import torch
from torch import nn
import numpy as np
import importlib

def get_coords(nx, range = (0,1), dim=2):
    """ Get coordinates for INR evaluation
    Args:
        nx (int): Number of pixels in one side of rasterization grid (assumes square image)
        K (int): Frequency cutoff. 

    Returns:
        mask (bool tensor): Mask indicating location of low-pass samples in FFT domain.
    """
    tensors = tuple(dim * [torch.linspace(range[0], range[1], steps=nx)])
    coords = torch.stack(torch.meshgrid(*tensors,indexing="ij"), dim=-1)
    coords = coords.reshape(-1, dim)
    return coords

# 
def get_fourier_sampling_mask(nx,K):
    """ Get binary mask to extract low-pass Fourier samples using FFT indexing
    Args:
        nx (int): Number of pixels in one side of rasterization grid (assumes square image)
        K (int): Frequency cutoff. 

    Returns:
        mask (bool tensor): Mask indicating location of low-pass samples in FFT domain.
    """
    freq = torch.fft.fftfreq(nx,d=1/nx)
    freq_x,freq_y = torch.meshgrid(freq,freq,indexing="ij")
    mask = (torch.max(torch.abs(freq_x),torch.abs(freq_y)) <= K)
    return mask

def get_inr(arch: str, arch_options: dict):
    """ Get a INR architecture
    Args:
        arch (str): The name of the Python file in the `arch` folder (without `.py`).
        arch_options (dict): Options to pass to the architecture builder.
        vars (dict): Additional variables used to the define the INR (optional)

    Returns:
        nn.Module: The instantiated PyTorch INR model.
    """
    try:
        module_path = f"arch.{arch}"
        arch_module = importlib.import_module(module_path)

        # The convention is that each arch module must expose a `build()` function
        if hasattr(arch_module, "build"):
            return arch_module.build(arch_options)
        else:
            raise AttributeError(f"Module '{arch}' does not define a 'build' function.")
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not find architecture '{arch}' in 'arch/' folder.") from e






