
# from torch import nn
# from torch.nn import functional as F
# from torch import optim


from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
import math

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

#####################################################################

class Conv2d(Module):
    """
    Class for implementing the Convolutional layer
    """

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):

        self.input_channels = input_channels
        self.output_channels = output_channels

        # if only an int size is give for kernel_size/stride/padding make a tuple from it        
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size 

        if type(stride) is int:
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if type(padding) is int:
            self.padding = (padding, padding)
        else:
            self.padding = padding


        k_sqrt = (1/(input_channels*self.kernel_size[0]*self.kernel_size[1])) ** .5
        self.weight = torch.empty((output_channels, input_channels, self.kernel_size[0], self.kernel_size[1])).uniform_(-k_sqrt, k_sqrt)
        self.bias = torch.empty(output_channels).uniform_(-k_sqrt, k_sqrt)

        # gradient tensors 
        self.weight_grad = torch.empty(self.weight.shape).zero_()
        self.bias_grad = torch.empty(self.bias.shape).zero_()

        # for backward pass computation
        self.input_x = None
        self.forward_output = None

    def forward(self, *input):

        def apply_conv(tensor):

            self.input_x = tensor

            in_unfolded = unfold(tensor, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            out_unfolded = self.weight.view(self.output_channels, -1) @ in_unfolded + self.bias.view(1, -1, 1)

            h_out = math.ceil((tensor.shape[2]+(2*self.padding[0]) - self.kernel_size[0])/self.stride[0] + 1)
            w_out = math.ceil((tensor.shape[3]+(2*self.padding[1]) - self.kernel_size[1])/self.stride[1] + 1)
            output = out_unfolded.view(tensor.shape[0], self.output_channels, h_out, w_out)
            
            return output

        # Apply forward function on either a single tensor or tuple of tensors
        # and stores values needed for backward
        if len(input) == 1:
            self.forward_output = apply_conv(input[0])
        else:
            self.forward_output = tuple(map(lambda tens: apply_conv(tens), input))

        return self.forward_output

    def backward(self, *gradwrtoutput):
        
        def backward_pass(gradwrtoutput):

            input_x_unfolded = unfold(self.input_x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

            self.weight_grad += (gradwrtoutput.view(self.output_channels, -1) @ input_x_unfolded.squeeze(0).t()).view(self.weight_grad.shape)
            self.bias_grad += gradwrtoutput.sum((0,2,3))

            gradwrtinput = self.weight.view(self.output_channels, -1).t() @ gradwrtoutput.view(1, self.output_channels, -1)
            gradwrtinput = fold(gradwrtinput, output_size=(self.input_x.shape[2], self.input_x.shape[3]), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            
            return gradwrtinput

        # Apply backward function as needed
        if len(gradwrtoutput) == 1:
            return backward_pass(gradwrtoutput[0])
        else:
            return tuple(map(lambda grad: backward_pass(grad), input))

    def param(self):
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]

###################################################################

class Upsampling(Module):
    """
    Class for implementing the Upsampling using the Transpose convolutional layer
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        # like for Conv2d
        self.input_channels = input_channels
        self.output_channels = output_channels
        # if only an int size is give for kernel_size/stride/padding make a tuple from it 
        if type(kernel_size) is int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size 

        if type(stride) is int:
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        if type(padding) is int:
            self.padding = (padding, padding)
        else:
            self.padding = padding

        k_sqrt = (1/(input_channels*self.kernel_size[0]*self.kernel_size[1])) ** .5
        self.weight = torch.empty((input_channels, output_channels, self.kernel_size[0], self.kernel_size[1])).uniform_(-k_sqrt, k_sqrt)
        self.bias = torch.empty(output_channels).uniform_(-k_sqrt, k_sqrt)

        # gradient tensors 
        self.weight_grad = torch.empty((input_channels, output_channels, self.kernel_size[0], self.kernel_size[1])).zero_()
        self.bias_grad = torch.empty(output_channels).zero_()

        # for backward pass computation
        self.input_x = None
        self.forward_output = None

    def forward(self, *input):
        
        def apply_conv(tensor):
            
            self.input_x = tensor

            linear_operation = (self.weight.view(self.input_channels, -1).t() @ tensor.view(self.input_channels, -1))

            h_out = (tensor.shape[2] - 1) * self.stride[0] - (2*self.padding[0]) + self.kernel_size[0]
            w_out = (tensor.shape[3] - 1) * self.stride[1] - (2*self.padding[1]) + self.kernel_size[1]

            folded = fold(linear_operation, output_size=(h_out, w_out), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
            output = folded.view(tensor.shape[0], folded.shape[0], folded.shape[1], folded.shape[2]) + self.bias.view(tensor.shape[0], self.output_channels, 1, 1)
            return output

        if len(input) == 1:
            self.forward_output = apply_conv(input[0])
        else:
            self.forward_output = tuple(map(lambda tens: apply_conv(tens), input))

        return self.forward_output

    def backward(self, *gradwrtoutput):
        
        def backward_pass(gradwrtoutput):

            gradwrtoutput_unfolded = unfold(gradwrtoutput, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

            self.weight_grad += (gradwrtoutput_unfolded @ self.input_x.view(self.input_channels, -1).t()).view(self.weight_grad.shape)
            self.bias_grad += gradwrtoutput.sum((0,2,3))

            gradwrtinput = (self.weight.view(self.input_channels, -1) @ gradwrtoutput_unfolded).view(self.input_x.shape)

            return gradwrtinput

        # Apply backward function as needed
        if len(gradwrtoutput) == 1:
            return backward_pass(gradwrtoutput[0])
        else:
            return tuple(map(lambda grad: backward_pass(grad), input))

    def param(self):
        return [(self.weight, self.weight_grad), (self.bias, self.bias_grad)]

########################################################################

class ReLU(Module):

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

#-------------------------------------------------------------------------------

class Sigmoid(Module):

    def forward(self, *input):

        # Forward function
        def forward(tens):
            output = tens
            output = 1/(1+((-output).exp()))
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

#--------------------------------------------------------------------------------

class Sequential(Module):
    """
    Class implementing our model with sequential layers
    """
    def __init__(self, *layers):
        # initializes the model with all the layers give as parameters
        self.layers = list(layers)

    def forward(self, *input):
        # performs the forward pass through the model

        x = input[0]
        for l in self.layers:
            x = l.forward(x)
        
        return x

    def backward(self, *gradwrtoutput):
        # performs the backward pass through the model
        
        x = gradwrtoutput[0]

        for l in reversed(self.layers):
            x = l.backward(x)

        return x

    def param(self):
        # return the parameters of all the layers
        parameters = []

        for l in self.layers:
            for p in l.param():
                parameters.append(p)

        return parameters

    def append_layer(self, layer):
        # appends a new layer to the model
        self.layers.append(layer)


# ---------------------------------------------------------------

class MSE(Module):
    def __init__(self):
        """
        MSE loss
        """
        self.y = None
        self.target = None
        self.e = None
        self.n = None

    def forward(self, y, target):
        """
        MSE computation
        :param y: output of the final layer, torch.Tensor
        :param target: target data, torch.Tensor
        :returns: MSE(f(x), y) = Sum(e^2) / n, e = y - f(x)
        """

        self.y = y.clone()
        self.target = target.clone()
        self.e = (self.y - self.target)
        self.n = self.y.size(0)

        return self.e.pow(2).mean()

    def backward(self):
        """
        MSE gradient computation
        :returns: Grad(MSE(f(x), y)) = 2e / n, e = y - f(x)
        """
        return 2 * self.e / self.n
    
    def param(self):
        #No parameters
        return []


# ---------------------------------------------------------------


class SGD(object):
    """
    Class implementing mini-batch SGD optimization
    """

    def __init__(self, model, nb_epochs=50, mini_batch_size=1, lr=1e-3, criterion=MSE()):
        """
        SGD constructor
        :param model: the model to train
        :param nb_epochs: maximum number of training epochs, positive int, optional, default is 50
        :param mini_batch_size: number of samples per mini-batch, int in [1, num_train_samples], optional, default is 1
        :param lr: learning rate, positive float, optional, default is 1e-2
        :param criterion: loss function to optimize, models.Module object, optional, default is criteria.LossMSE
        """
        
        if not isinstance(nb_epochs, int) or nb_epochs <= 0:
            raise ValueError("Number of training epochs must be a positive integer")
        if not isinstance(mini_batch_size, int) or mini_batch_size <= 0:
            raise ValueError("Mini-batch size must be a positive integer")
        if not isinstance(lr, float) or lr <= 0:
            raise ValueError("Learning rate must be a positive number")
        
        self.model = model
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        self.criterion = criterion

    def step(self):
        a=self.model.param()
        a[0][0].sub_(self.lr*a[0][1])
        a[1][0].sub_(self.lr*a[1][1])
        
    def train(self,train_input,train_target):
        """
        Function implementing the mini-batch training procedure
        :param train_input: torch.Tensor with train input data
        :param train_target: torch.Tensor with train target data
        :returns: the trained model
        """

        for e in range(self.nb_epochs):
            sum_loss = 0.

            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.model.forward(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion.forward(output, train_target.narrow(0, b, self.mini_batch_size))

                sum_loss += loss

                l_grad = self.criterion.backward()
                self.model.backward(l_grad)
                self.step()

            print("{} iteration: loss={}".format(e, sum_loss))
            
        return self.model



# ===============================================================


class Model ():
    def __init__(self) -> None :
        
        self.model = Sequential(
            Conv2d(3, 64, 3, stride=2), 
            ReLU(), 
            Conv2d(64, 48, 3, stride=2), 
            ReLU(), 
            Upsampling(48, 64, 3),
            ReLU(),
            Upsampling(64, 3, 3), 
            Sigmoid())
        #self.model.to(device)

        self.loss = MSE()
        #self.loss.to(device)

    def load_pretrained_model(self) -> None :
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        mini_batch_size = 10

        #train_input.to(device)
        #train_target.to(device)

        sgd = SGD(model=self.model, nb_epochs = num_epochs, mini_batch_size=mini_batch_size, criterion=self.loss)
        self.model = sgd.train(train_input, train_target)


    def predict(self, test_input ) -> torch.Tensor:
        #:test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        #test_input.to(device)

        return self.model.forward(test_input)

