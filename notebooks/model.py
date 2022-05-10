# -*- coding: utf-8 -*-
# %%
import torch
import torch.nn as nn
from torch import optim

# %% [markdown]
# #### Load data

# %%
device = torch . device ('cuda' if torch.cuda.is_available() else 'cpu')

# %%
noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl')
noisy_imgs, clean_imgs = torch.load('../data/val_data.pkl')

# %%
noisy_imgs_1.shape


# %% [markdown]
# **Evaluation metric**

# %%
def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio: denoised and ground-truth have range [0,1]
    mse = torch.mean((denoised - ground_truth)**2)
    return (-10)*torch.log10(mse + (10**(-8)))


# %%
### For mini - project 1
class Model ():
    def __init__(self) -> None :
        # instantaiate model + optimizer + loss function + any other stuff you need

        # model: UNet
        self.model = nn.Sequential()
        
        # optimizer: Adam; try different learning rates
        self.optim = optim.Adam(self.model.parameters(), lr = 1e-1)

        # loss function: MSE or HDRLoss for MonteCarlo images ? 
        self.loss = nn.MSELoss()

    def load_pretrained_model(self) -> None :
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images

        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        pass

    def predict(self, test_input ) -> torch.Tensor:
        #:test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        pass
