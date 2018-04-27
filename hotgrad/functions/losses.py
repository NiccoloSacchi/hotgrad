# -*- coding: utf-8 -*-
""" Implementation of the losses. """

from hotgrad.module import Module
import hotgrad.functions.operands
from hotgrad.module import Module2Operands

class MSE(Module2Operands):
    """
    Computes the Mean Squared Error
    """
    def __init__(self):
        super(MSE, self).__init__()
        
    def clear(self):
        self.__init__()

    def __call__(self, input, target):
        return self.forward(input, target)
        
    # TODO the module should remember who input and target. Just Variables should remember which operations were performed
    def forward(self, input, target):
        # no need to explicitly create variable. Right?
        super(MSE, self).forward(input, target)
        assert self.l_input.data.shape == self.r_input.data.shape, "Broadcasting is not supported" # for simplicity 
        
        return (self.r_input.sub(self.l_input)).pow(2).mean()
    
    def backward(self, grad):
        l_input_grad = 2*(self.l_input - self.r_input).mean() * grad
        r_input_grad = -l_input_grad
        
        self.l_input.backward(grad = l_input_grad)
        self.r_input.backward(grad = r_input_grad)
