
# from torch import nn
# from torch.nn import functional as F
# from torch import optim


from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

import torch
# torch.cuda.get_device_name(torch.cuda.current_device())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===============================================================

class Conv2d(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ConvTranspose2d(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Upsample(object):

    ## Just use ConvTranspose2d I would say. No need to implement anything new..

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sigmoid(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


# ---------------------------------------------------------------

class MSE(object):
    pass


# ---------------------------------------------------------------


class SGD(object):
    pass



# ===============================================================


class Model ():
    def __init__(self) -> None :
        # instantaiate model + optimizer + loss function + any other stuff you need

        # model: UNet
        # optimizer: Adam?
        # loss function: MSE or HDRLoss for MonteCarlo images ?
        pass

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

