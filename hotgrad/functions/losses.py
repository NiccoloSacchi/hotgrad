# -*- coding: utf-8 -*-
""" Implementation of the losses. """

from hotgrad.module import Module

class MSE(Module):
    """
    Computes the Mean Squared Error:
        formula
    """
    
    # TODO the module should remember who input and target. Just Variables should remember which operations were performed
    def forward(self, input, target):
        self.input = input
        self.target = target
        return (target - input).pow(2).mean(0)
    
    def MSE_derivative(self, input, target):
        return 2*(input - target).mean(0)
    
    def backward(self):
        dl_dx = self.MSE_derivative(self.input, self.target)
        
