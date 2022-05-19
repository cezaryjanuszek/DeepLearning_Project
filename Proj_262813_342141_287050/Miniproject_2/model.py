
# from torch import nn
# from torch.nn import functional as F
# from torch import optim


from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

import torch
# torch.cuda.get_device_name(torch.cuda.current_device())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.set_grad_enabled(False)

# ===============================================================

# Module class that the others will inherit from
class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Conv2d(Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding

        k_sqrt = (1/(input_channels*kernel_size[0]*kernel_size[1])) ** .5
        self.weight = torch.empty((output_channels, input_channels, kernel_size[0], kernel_size[1])).uniform_(-k_sqrt, k_sqrt)
        self.bias = torch.empty(output_channels).uniform_(-k_sqrt, k_sqrt)

    def forward(self, *input):

        def apply_conv(tensor):
            in_unfolded = unfold(tensor, self.kernel_size)
            out_unfolded = self.weight.view(self.weight.size(0), -1) @ in_unfolded + self.bias.view(1, -1, 1)
            ouput = out_unfolded.view(1, self.output_channels, tensor.shape[2]-self.kernel_size[0]+1, tensor.shape[3]-self.kernel_size[1]+1)
            return ouput

        # Apply forward function on either a single tensor or tuple of tensors
        # and stores values needed for backward
        if len(input) == 1:
            self.forward_output = apply_conv(input[0])
        else:
            self.forward_output = tuple(map(lambda tens: apply_conv(tens), input))

        return self.forward_output

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class TransposeConv2d(Module):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Upsample(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(object):
    def forward(self, *input):

        # Forward function
        def forward(tens):
            output = tens
            output[output < 0] = 0
            return output

        # Apply forward function on either a single tensor or tuple of tensors
        # and stores values needed for backward
        if len(input) == 1:
            self.forward_output = forward(input[0])
        else:
            self.forward_output = tuple(map(lambda tens: forward(tens), input))

        return self.forward_output

    def backward(self, *gradwrtoutput):

        # Backward function
        def backward(forward_output, gradwrtoutput):

            # gradient of ReLU function
            grad = forward_output
            grad[grad > 0] = 1
            return grad * gradwrtoutput

        # Apply backward function as needed
        if len(gradwrtoutput) == 1:
            return backward(self.forward_output, gradwrtoutput[0])
        else:
            bkw = []
            for (forward_output, gradwrtoutput) in zip(self.forward_output, gradwrtoutput):
                bkw.append(backward(forward_output, gradwrtoutput))
            return tuple(bkw)

    def param(self):
        # No parameters
        return []


# TODO Do we leave e hardcoded or use math.exp or smth?
E = 2.7182818

class Sigmoid(object):

    def forward(self, *input):

        # Forward function
        def forward(tens):
            output = tens
            output = 1/(1+(E ** (-output)))
            return output

        # Apply forward function on either a single tensor or tuple of tensors
        # and stores values needed for backward
        if len(input) == 1:
            self.forward_output = forward(input[0])
        else:
            self.forward_output = tuple(map(lambda tens: forward(tens), input))

        return self.forward_output

    def backward(self, *gradwrtoutput):

        # Backward function
        def backward(forward_output, gradwrtoutput):

            # gradient of Sigmoid function
            grad = 1 - forward_output
            return grad * gradwrtoutput

        # Apply backward function as needed
        if len(gradwrtoutput) == 1:
            return backward(self.forward_output, gradwrtoutput[0])
        else:
            bkw = []
            for (forward_output, gradwrtoutput) in zip(self.forward_output, gradwrtoutput):
                bkw.append(backward(forward_output, gradwrtoutput))
            return tuple(bkw)

    def param(self):
        # No parameters
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

