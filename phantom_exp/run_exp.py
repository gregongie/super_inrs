import argparse
import importlib

import torch
import numpy as np
from PIL import Image
import json

from utils import get_fourier_sampling_mask, get_coords, get_inr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help="Name of the experiment settings file (without .py)")
    parser.add_argument('--device', required=False, help="Device to use: 'cuda:0', 'cuda:1', etc, or 'cpu'. Defaults to GPU 0 if available.")
    args = parser.parse_args()

    # Set the device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load settings from specified file in exp subfolder
    settings_module = importlib.import_module(f"exp.{args.exp}")
    settings = settings_module.settings

    print(f"\nRunning experiment: {args.exp}")

    print("\nLoaded settings:")
    for k, v in settings.items():
        print(f"{k}: {v}")

    inr, x, metrics = run_experiment(settings,device)

    print(f"\nFinished! Saving outputs.")

    #save inr weights
    torch.save(inr.state_dict(), f"results/{args.exp}.pth") 

    #save rasterized inr output image (x)
    np.save(f"results/{args.exp}.npy", x) 

    #save final metrics as JSON file
    with open(f"results/{args.exp}.json", "w") as f:         
        json.dump(metrics, f, indent=4)

    #also save rasterized output image as .png for easy visualization 
    # note: this is a lossy conversion
    img = (np.clip(x,0,1) * 255).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(f"results/{args.exp}.png")               

    print(f"\nDone.")

def run_experiment(settings,device):
    #load data
    phantom_name = settings["phantom_name"]
    x0 = np.load("data/"+phantom_name+"_lowpass_1024.npy") #ideal low-pass version of phantom computed from *exact* fourier coefficients
    x00 = np.load("data/"+phantom_name+"_rasterized_1024.npy") #hi-res rasterized phantom for reference
    x0 = torch.from_numpy(x0).float().to(device)
    x00 = torch.from_numpy(x00).float().to(device)

    # define fourier sampling mask
    K = settings["K"] #sampling frequency cutoff
    nx = settings["nx"] #recon grid size
    res = (nx,nx) #image resolution over which to perform FFTs
    mask = get_fourier_sampling_mask(nx,K)

    # get measurement vector y
    y = torch.fft.fft2(x0,norm="ortho")[mask]  #low-pass Fourier coefficients

    # define MSE metric
    MSE = torch.nn.MSELoss()

    # compute zero-filled ifft recon as a baseline
    x1 = torch.zeros(res, dtype=torch.complex64, device=device)
    x1[mask] = y
    x1 = torch.real(torch.fft.ifft2(x1,norm="ortho"))
    initmse = MSE(x1,x00)
    print(f"MSE of zero-filled IFFT: {initmse.item():>4e}")

    # get INR coordinates
    coords_range = settings["coords_range"]
    coords = get_coords(nx, range = coords_range, dim=2)
    coords = coords.to(device)

    # problem-specific variables to pass to INR as needed
    vars = {"coords": coords, "res": res, "mask": mask}

    # define INR architecture
    torch.manual_seed(settings["seed"]) #set seed for reproducibility
    inr = get_inr(settings["arch"],settings["arch_options"])
    if "arch_init_wts" in settings: #apply custom initialization if specificed
        inr.load_state_dict(torch.load(f"wts/{settings["arch_init_wts"]}", weights_only=True))
        inr.eval()
    inr.register(vars) #register extra vars if needed (mainly for ffrelu_shallow INR with modified WD-reg)
    inr = inr.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(inr.parameters(),lr=settings["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=settings["step_size"],gamma=settings["gamma"])

    lam = settings["lambda"]
    epochs = settings["epochs"]

    # run training loop
    for iter in range(epochs):
        optimizer.zero_grad()

        x = inr(coords).view(res)
        Ax = torch.fft.fft2(x,norm="ortho")[mask]
        mseloss = MSE(torch.real(Ax),torch.real(y))+MSE(torch.imag(Ax),torch.imag(y))
        wd_reg = inr.weight_decay()
        loss = mseloss + lam*wd_reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        # print metrics
        if iter % 100 == 99:
            with torch.no_grad():
                imgmse = MSE(x,x00)  #MSE with ground truth rasterized phantom
                print(f"iter: {iter+1}, loss: {loss.item():>4e}, DataMSE: {mseloss.item():>4e}, WDReg: {wd_reg.item():>4e}, ImgMSE: {imgmse.item():>4e}")

    #compute final metrics
    with torch.no_grad():
        x = inr(coords).view(res)
        Ax = torch.fft.fft2(x,norm="ortho")[mask]
        mseloss = MSE(torch.real(Ax),torch.real(y))+MSE(torch.imag(Ax),torch.imag(y))
        wd_reg = inr.weight_decay()
        loss = mseloss + lam*wd_reg
        imgmse = MSE(x,x00)

    print("final metrics:\n")
    print(f"loss: {loss.item():>4e}, DataMSE: {mseloss.item():>4e}, WDReg: {wd_reg.item():>4e}, ImgMSE: {imgmse.item():>4e}")

    x = x.cpu().numpy()
    metrics = {}
    metrics["final_loss"] = loss.item()
    metrics["final_mseloss"] = mseloss.item()
    metrics["final_wd_reg"] = wd_reg.item()
    metrics["final_imgmse"] = imgmse.item()
    
    return inr, x, metrics

if __name__ == "__main__":
    main()
