# -*- coding: utf-8 -*-
""" Here are define the base operand that can be applied on Variables so to keep the acyclic graph
and allow the backpropagation of the gradient. """

import operator
from functools import reduce
from torch import FloatTensor, is_tensor
import math

import hotgrad

from hotgrad.module import Module2Operands, Module1Operand

class Add(hotgrad.module.Module2Operands):
    def __init__(self, l_input, r_input):
        super(Add, self).__init__(l_input, r_input)
        
        assert self.l_input.data.shape == self.r_input.data.shape, "Broadcasting is not supported" # for simplicity 
        return
    
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(self.l_input.data + self.r_input.data, previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = r_grad = 1 * grad
        
        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
        
class Sub(hotgrad.module.Module2Operands):
    def __init__(self, l_input, r_input):
        super(Sub, self).__init__(l_input, r_input)
        
        assert self.l_input.data.shape == self.r_input.data.shape, "Broadcasting is not supported" # for simplicity 
        return
    
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(self.l_input.data - self.r_input.data, previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = 1 * grad
        r_grad = (-1) * grad
        
        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)

class Mul(hotgrad.module.Module2Operands):
    def __init__(self, l_input, r_input):
        super(Mul, self).__init__(l_input, r_input)
        
        assert self.l_input.data.shape == self.r_input.data.shape, "Broadcasting is not supported" # for simplicity 
        return
        
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(self.l_input.data*self.r_input.data, previous_op=self)
        
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = self.r_input.data * grad
        r_grad = self.l_input.data * grad

        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
        return
    
class MatMul(hotgrad.module.Module2Operands):
    def __init__(self, l_input, r_input):
        super(MatMul, self).__init__(l_input, r_input)
        
        assert (self.l_input.data.dim()<=2) or (self.r_input.data.dim()<=2), "Maximum supoorted dimension is 2"
        assert self.l_input.data.shape[-1] == self.r_input.data.shape[0], ("The Variable shape does not allow matrix multiplication: " + str(self.l_input.data.shape) + " @ " + str(self.r_input.data.shape))
        return
    
    def forward(self):
        """ Compute the forward pass. """        
        result = self.l_input.data @ self.r_input.data
        if not is_tensor(result): # make sure you have a tensor (not a float/int)
            result = FloatTensor([result])
        return hotgrad.variable.Variable(result, previous_op=self)
        
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        # first transpose the two input then do the matrix multiplication with the received gradient
        r_input_t = self.r_input.data.t() if self.r_input.data.dim()==2 else self.r_input
        l_input_t = self.l_input.data.t() if self.l_input.data.dim()==2 else self.l_input
        
        l_grad = grad @ r_input_t
        r_grad = l_input_t @ grad
        # if both an input and grad had shape (1,) then the output is just a float: convert back to FloatTensor
        if not is_tensor(l_grad):
            l_grad = FloatTensor([l_grad])
        if not is_tensor(r_grad):
            r_grad = FloatTensor([r_grad])

        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
        return
    
class Pow(hotgrad.module.Module2Operands):
    def __init__(self, l_input, r_input):
        if isinstance(r_input, int):
            r_input = hotgrad.variable.Variable(FloatTensor([r_input]))
        assert (isinstance(r_input, int) or l_input.data.shape == r_input.data.shape or r_input.shape == (1,)), "r_input must have the same shape as the l_input or must have shape (1,)"

        super(Pow, self).__init__(l_input, r_input)
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(self.l_input.data.pow(self.r_input.data), previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = grad * (self.r_input.data * self.l_input.data.pow(self.r_input.data - 1))
        #r_grad = self.l_input.data.log() * self.l_input.data.pow(self.r_input.data)
        
        self.l_input.backward(grad = l_grad)
        return

class Mean(Module1Operand):
    def forward(self):
        """ Compute the forward pass. """
        return hotgrad.variable.Variable(FloatTensor([self.input.data.mean()]), previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the input Variable. """
        if grad.shape != (1,):
            raise BackwardException("The Mean module must receive a gradient of shape (1,)")
        
        num_entries = reduce(operator.mul, list(self.input.shape))
        # compute the gradient wrt the input
        input_grad = FloatTensor(self.input.shape).fill_(1/num_entries)
        self.input.backward(grad = input_grad*grad)
        return
