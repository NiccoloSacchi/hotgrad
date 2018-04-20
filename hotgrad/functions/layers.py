# -*- coding: utf-8 -*-
""" Implementation of the layers. """

from hotgrad.module import Module
from torch import Tensor

class Linear(Module):
    """
    Implements a fully connected layer
    """
    
    def __init__(self, input_features, output_features, bias=None):
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias
        self.weight = Tensor(input_features, output_features)
    
    def forward(self, input):
        # input: N x input_features matrix
        # weigth: input_features x output_features matrix
        # returns N x output_features matrix
        res = input.mm(Tensor(self.weight))
        if self.bias != None:
            res += self.bias
        return res
    
    # TODO: need to interact with the optimizer
    def backward(self, dl_ds_next_layer):
        # input: N x input_features matrix
        # dl_ds: N x output_features matrix
        self.grad_w = self.input.t().mm(dl_ds_next_layer)
        self.grad_b = dl_ds_next_layer
        
    
