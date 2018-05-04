# -*- coding: utf-8 -*-
""" Implementation of the layers. """

from hotgrad.module import Module
from torch import FloatTensor
import hotgrad

class Linear():
    """
    Implements a fully connected layer
    """
    
    def __init__(self, output_features):
        self.output_features = output_features
        self.bias = hotgrad.variable.Variable(FloatTensor(1, output_features).normal_(0), requires_grad=True)
        self.weight = None
        
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        if self.weight == None:
            input_features = input.shape[1]
            self.input_features = input_features
            a = 1/(input_features**0.5)
            self.weight = hotgrad.variable.Variable(FloatTensor(self.input_features, self.output_features).uniform_(-a, a), requires_grad=True)

        return (input @ self.weight) + self.bias
        
    def params(self):
        return [self.weight, self.bias]

    def clear(self):
        self.__init__(self.output_features)