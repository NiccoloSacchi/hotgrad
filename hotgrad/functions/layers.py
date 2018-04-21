# -*- coding: utf-8 -*-
""" Implementation of the layers. """

from hotgrad.module import Module
from torch import FloatTensor
import hotgrad

class Linear():
    """
    Implements a fully connected layer
    """
    
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        self.weight = hotgrad.variable.Variable(FloatTensor(input_features, output_features).normal_(0), requires_grad=True)
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        return input @ self.weight
        
    def params(self):
        return self.weight
