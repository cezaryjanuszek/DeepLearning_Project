
from torch import nn
from torch.nn import functional as F
from torch import optim
from pathlib import Path

import torch
# torch.cuda.get_device_name(torch.cuda.current_device())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model():
    def __init__(self) -> None :
        # instantaiate model + optimizer + loss function + any other stuff you need

        # TODO insert best model we have in here

        # model
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.ReLU(inplace=1),
            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(-0.1)
        )
        self.model.to(device)

        # optimizer: Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)

        # loss function: MSE
        self.loss = nn.MSELoss()
        self.loss.to(device)

    def load_pretrained_model(self) -> None :
        ## This loads the parameters saved in bestmodel.pth into the model
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.model = torch.load(model_path)
        self.model.to(device)


    def train(self, train_input, train_target, num_epochs) -> None :
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        mini_batch_size = 100

        train_input=train_input.float().to(device)
        train_target=train_target.float().to(device)

        print('--------------------------')
        print('Training model')

        for e in range(num_epochs):
            print('Epoch:', e)
            losses = []
            for b in range(0, train_input.size(0), mini_batch_size):
                output = self.model(train_input.narrow(0, b, mini_batch_size))
                loss = self.loss(output, train_target.narrow(0, b, mini_batch_size))
                loss.requires_grad_()
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('Loss = ', sum(losses)/(train_input.size(0)/mini_batch_size))

        print('---------------------------')

    def predict(self, test_input) -> torch.Tensor:
        #:test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size (N1 , C, H, W)
        test_input=test_input.float().to(device)

        return self.model(test_input)



