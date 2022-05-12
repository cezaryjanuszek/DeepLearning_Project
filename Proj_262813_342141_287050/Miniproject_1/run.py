
from . import model
import torch

noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl')
train_input, train_target = noisy_imgs_1.float()/255.0, noisy_imgs_2.float()/255.0

model = model.Model()
nb_epochs = 30

model.train(train_input, train_target, nb_epochs)

def save_model(model):
    torch.save(model, 'bestmodel.pth')

print('Saving model')
save_model(model.model)

