import torch
from torch import nn
import numpy as np
from timeit import default_timer as timer

# set gpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#############################################
# helper functions to define INR architecture
#############################################

# define single relu layer
def relu_layer(input_dim,width):
    layers = [nn.Linear(input_dim, width, bias=False), nn.ReLU()]
    return nn.Sequential(*layers)

# Fourier feature mapping
def ff_mapping(x, B):
    #x is a vector of coordinates, size=[input_dim]
    #B is the frequencies matrix, size=[#freq,input_dim]
    x_proj = (2.0*np.pi*x) @ B.T
    # return torch.cat([torch.sin(x_proj), torch.cos(x_proj), axis=-1)
    return torch.cat([np.sqrt(2)*torch.sin(x_proj), np.sqrt(2)*torch.cos(x_proj)], axis=-1) #note: no constant feature in this case

# get coordinates for evaluation of the CBNN
def get_coords(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of 0 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

B = torch.eye(2) #use restricted Fourier features only
B = B.to(device)
n = B.shape[0] #n frequency pairs -> 2n+1 fourier features (where +1 is coming from constant feature)
nx=1024
res = (nx,nx)
coords = get_coords(nx, dim=2).to(device) #tensor of coordinates to evaluate CBNN at

#############################################
# define teacher network
#############################################
def teacher_net(W,random_seed):
  torch.manual_seed(random_seed)
  input_dim = 2*n+1 #+1 for additional constant feature
  rlayer = relu_layer(input_dim,W)
  avec = torch.empty(W)
  for i in range(W):
    w = 2*torch.rand(1,input_dim)-1
    w = w/torch.norm(w)
    rlayer[0].weight.data[i] = w.clone()
    a = torch.rand(1).item() * 4 + 1 #restrict to positive amplitudes
    avec[i] = a

  rlayer = rlayer.to(device)
  avec = avec.to(device)
  teacher_network = lambda x : rlayer(ff_mapping(x,B))
  with torch.no_grad():
    x0 =  (avec@(teacher_network(coords).T)).view(res)
    x0 = (x0 - x0.min()) / (x0.max() - x0.min()) #normalizing x0
  return x0
##############################

#############################################
# define student network
#############################################
def student_net(random_seed):
  torch.manual_seed(random_seed)
  input_dim = 2*n
  width_s = 100

  #define initial relu layer
  rlayer = relu_layer(input_dim,width_s)
  rlayer = rlayer.to(device)

  #define outer layer weight vector
  avec = torch.empty(width_s)
  nn.init.uniform_(avec,-1,1) #uniform -1, 1 weights 
  avec = avec/10 #rescale avec so that CBNN output is roughly in range 0 to 1
  avec = avec.to(device)
  avec = avec.requires_grad_()

  return rlayer, avec, width_s
#####################################

#fourier measurements
def generate_fourier_measurement(K,x0):
  freq = torch.fft.fftfreq(nx,d=1/nx)
  freq_x,freq_y = torch.meshgrid(freq,freq)
  mask = (torch.max(torch.abs(freq_x),torch.abs(freq_y)) <= K)
  mask = mask.to(device)

  y = torch.fft.fft2(x0,norm="forward")[mask]
  x1 = torch.zeros(res, dtype=torch.complex64, device=device)
  x1[mask] = y
  x1 = torch.real(torch.fft.ifft2(x1,norm="forward"))

  return  y, mask, x1

###########################################
### train the student
def train_student(rlayer, avec, width_s, mask, y, param, std_wd=True):
  loss_values=[]
  mseloss_values = []
  imgloss_values=[]
  imglossrel_values=[]
  maxabserr_values =[]
  min_imgloss= float('inf')
  min_maxabserr= float('inf')
  count=0
  Flag=False #early termination flag

  yreal = torch.real(y) #separate out real an imaginary parts to facilitate MSE computation
  yimag = torch.imag(y)


  lamreal = torch.zeros_like(yreal) #lagrange multipliers
  lamimag = torch.zeros_like(yimag)

  MSE = torch.nn.MSELoss() #mean-squared error loss function

  #hyper-parameters
  mu_init = param["mu_init"]
  mu_inc_factor = param["mu_inc_factor"]
  outer_iters = param["outer_iters"]
  inner_iters = param["inner_iters"]
  lr = param["lr"]

  hidden = lambda x: rlayer(ff_mapping(x,B))
  optimizer = torch.optim.Adam(list(rlayer.parameters())+list([avec]),lr=lr)

  mu = mu_init
  for k in range(outer_iters): #outer loop for updating lagrange multiplier/increasing mu
    if Flag:
      break

    for iter in range(inner_iters):
      hout = hidden(coords).T
      x = (avec@hout).view(res)
      Ax = torch.fft.fft2(x,norm="forward")[mask]
      Axreal = torch.real(Ax) #separate real and imaginary parts
      Aximag = torch.imag(Ax)

      if std_wd:
        wdloss = 0.5*((rlayer[0].weight**2).sum() + (avec**2).sum())
      else:
        wdloss = 0.5*((torch.sum(hout,dim=1)**2).sum()/(nx**4) + (avec**2).sum())

      mseloss = 0.5*((Axreal-yreal)**2 + (Aximag-yimag)**2).sum()
      lagrange = ((Axreal-yreal)*lamreal + (Aximag-yimag)*lamimag).sum()
      loss = wdloss + mu*mseloss + lagrange

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()


      mseloss_values.append(mseloss.item())
      loss_values.append(loss.item())

      # compute metrics
      with torch.no_grad():

        x = (avec@(hidden(coords).T)).view(res) 
        imgloss = MSE(x,x0)
        imglossrel = MSE(x,x0)/MSE(x1,x0)
        maxabserr = torch.max(torch.abs(x-x0))
        if std_wd:
          virtual_loss = (torch.norm(rlayer[0].weight,dim=1)*torch.abs(avec)).sum()
        else:
          virtual_loss = (torch.sum(hout,dim=1)*torch.abs(avec)).sum()/(nx**2)

        imgloss_values.append(imgloss.item())
        imglossrel_values.append(imglossrel.item())
        maxabserr_values.append(maxabserr.item())

        count = count+1
        if imgloss<min_imgloss: #smallest image loss dutring training
          min_imgloss=imgloss

        if maxabserr<min_maxabserr: #smallest max abs error during training
          min_maxabserr=maxabserr

        if min_imgloss.item()<=1e-9:
          Flag=True
          break


    # langrange multiplier update
    with torch.no_grad():
      lamreal = lamreal + mu*(Axreal-yreal)
      lamimag = lamimag + mu*(Aximag-yimag)

    #increase mu by constant multiplicative factor
    mu = mu_inc_factor*mu


  ################################


  print(f"iter: {count}, AL loss: {loss.item():>4e}, weight decay: {wdloss.item():>4e}, virtual loss: {virtual_loss.item():>4e}, DataMSE: {mseloss.item():>4e}, ImgMSE: {imgloss.item():>4e}, MinImgMSE: {min_imgloss.item():>4e}, RelMSE = {imglossrel.item():>4e}, MaxAbsErr = {maxabserr.item():>4e}, MinError = {min_maxabserr.item():>4e},mu: {mu}")

    # langrange multiplier update
  
  return imgloss.item(),min_imgloss.item(), imglossrel.item(), maxabserr.item(), min_maxabserr.item(), imgloss_values, loss_values, mseloss_values, imglossrel_values, maxabserr_values

  
#############################################################
# define the exact recovery occurs/fails cutoff
def exact_recovery(min_imgloss):
  if min_imgloss<=1e-9:
    return "OCCURS"
  else:
    return "FAILS"
############################################


# run the experiments:
ex_result = []

# Define the width and K values
width_values = [1,2,3,4,5,6,7,8,9,10]
k_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

job_id = 1 #replace with actual job_id when running on cluster; used as random seed to define phantom

# generating results
for W in (width_values):
    for K in (k_values):
        print(f'RESULTS FOR W: {W} & K: {K} & job:{job_id}')
        param = {"mu_init": 10, "mu_inc_factor": 1.1, "outer_iters": 100, "inner_iters": 5000, "lr": 0.001}
        x0 = teacher_net(W,job_id)
        y, mask, x1 = generate_fourier_measurement(K, x0)
        rlayer, avec, width_s = student_net(job_id)
        imgloss,min_imgloss, imglossrel, maxabserr, min_maxabserr, imgloss_values, loss_values, mseloss_values, imglossrel_values, maxabserr_values = train_student(rlayer, avec, width_s, mask, y, param, std_wd=True)
        output_result = exact_recovery(min_imgloss)
        x1_MaxAbsErr = torch.max(torch.abs(x1-x0))
        x1_MSE = torch.nn.functional.mse_loss(x1, x0)

        # store the results in a dictionary
        result_dict = {
            'W': W,
            'K': K,
            'job ID':job_id,
            'output': output_result,
            'Img loss': imgloss,
            'Min Img loss': min_imgloss,
            'Img loss rel': imglossrel,
            'Max abs err': maxabserr,
            'smallest Max err': min_maxabserr,
            'x1 MaxAbsErr': x1_MaxAbsErr,
            'x1 MSE':x1_MSE,

        }

print(result_dict)