import torch

from nn_model import nn_model

# Torch settings
torch.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Model(nn_model):

    def __init__(self, num_classes, steps, session, restore):
        return

    def run_model_train(self, batch_x, batch_y, batch_target_x, batch_target_y, batch_seq, session):
        return

    def run_model_test(self, batch_x, batch_y, batch_seq):
        return

    def save_model(self):
        return
