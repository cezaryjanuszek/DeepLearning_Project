
from .. import model
import torch
import pickle

noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pickle')
train_input, train_target = noisy_imgs_1.float(), noisy_imgs_2.float()

model = model.Model()
nb_epochs = 50

model.train(train_input, train_target, nb_epochs)

def save_model(model):
    pickle.dump(model, open('bestmodel.pkl', 'wb'))

print('Saving model')
save_model(model.model)