from module import Module
from torch import Tensor
from math import exp

class Tanh(Module):
    """
    Implements a fully connected layer
    """
    
    def forward(self, input):
        self.input = input
        res = (input.exp() - (-input).exp()) / (input.exp() + (-input).exp())
        return res
    
    # TODO: define an activation interface that is implemented by loss and relu?
    def tanh_gradient(self, input):
        return 1 - self.forward(input)
        
    # TODO: need to interact with the optimizer
    def backward(self, weights_next_layer, dl_ds_next_layer):
        # dl_ds_next_layer: N x output_features matrix
        # weights_next_layer: input_features x output_features matrix
        # dl_dx: N x input_features matrix
        self.dl_dx = dl_ds_next_layer.mm(weights_next_layer.t())
        self.dl_ds = self.dl_dx * self.tanh_gradient(self.input)
    