# -*- coding: utf-8 -*-
""" Implementation of the activation functions. """

from .module import Module
from torch import Tensor
from math import exp

class ReLU(Module):
    """
    Implements a rectified linear unit (ReLU) activation function.
    """
    
    def forward(self, input):
        self.input = input
        res = input.clone()
        res[res<0] = 0
        return res
    
    # TODO: define an activation interface that is implemented by loss and relu?
    def relu_gradient(self, input):
        res = Tensor(input.size()).fill_(1)
        res[res<0] = 0
        return res
        
    # TODO: need to interact with the optimizer
    def backward(self, weights_next_layer, dl_ds_next_layer):
        # dl_ds_next_layer: N x output_features matrix
        # weights_next_layer: input_features x output_features matrix
        # dl_dx: N x input_features matrix
        self.dl_dx = dl_ds_next_layer.mm(weights_next_layer.t())
        self.dl_ds = self.dl_dx * self.relu_gradient(self.input)
        
class Tanh(Module):
    """
    Implements Tanh activation function.
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
    
