# -*- coding: utf-8 -*-
""" Here are define the base operand that can be applied on Variables so to keep the acyclic graph
and allow the backpropagation of the gradient. """

import operator 
from functools import reduce
from torch import FloatTensor, is_tensor

import hotgrad

from hotgrad.module import Module2Operands, Module1Operand

class Mul(hotgrad.module.Module2Operands):
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(self.l_input.data*self.r_input.data, previous_op=self)
        
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = self.r_input.data * grad
        r_grad = self.l_input.data * grad
        # if both an input and grad had shape (1,) then the output is just a float: convert back to FloatTensor
        if not is_tensor(l_grad):
            l_grad = FloatTensor([l_grad])
        if not is_tensor(r_grad):
            r_grad = FloatTensor([r_grad])

        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
        return 

class Mean(Module1Operand):
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(FloatTensor([self.input.data.mean()]), previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        if grad.shape != (1,):
            raise BackwardException("The Mean module must receive a gradient of shape (1,)")
        
        num_entries = reduce(operator.mul, list(self.input.shape))
        # compute the gradient wrt the input
        input_grad = FloatTensor(self.input.shape).fill_(1/num_entries)
        self.input.backward(grad = input_grad*grad)
        return
