# -*- coding: utf-8 -*-
""" Here are define the base operand that can be applied on Variables so to keep the acyclic graph
and allow the backpropagation of the gradient. """

from torch import FloatTensor
import hotgrad
from hotgrad.module import Module2Operands, Module1Operand

class Add(hotgrad.module.Module2Operands):    
    def forward(self, l_input, r_input):
        """ Compute the forward pass. """        
        super(Add, self).forward(l_input, r_input)
        
        return hotgrad.variable.Variable(l_input.data + r_input.data, previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = r_grad = 1 * grad
        
        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
        
class Sub(hotgrad.module.Module2Operands):    
    def forward(self, l_input, r_input):
        """ Compute the forward pass. """        
        super(Sub, self).forward(l_input, r_input)
            
        return hotgrad.variable.Variable(l_input.data - r_input.data, previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = 1 * grad
        r_grad = (-1) * grad
        
        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)

class Mul(hotgrad.module.Module2Operands):
    def forward(self, l_input, r_input):
        """ Compute the forward pass. """                
        super(Mul, self).forward(l_input, r_input)
        
        return hotgrad.variable.Variable(l_input.data*r_input.data, previous_op=self)
        
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        l_grad = self.r_input.data * grad
        r_grad = self.l_input.data * grad

        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
    
class MatMul(hotgrad.module.Module2Operands):
    def forward(self, l_input, r_input):
        """ Compute the forward pass. """         
        assert (l_input.data.dim()<=2) or (r_input.data.dim()<=2), "Maximum supported dimension is 2"
        assert l_input.data.shape[-1] == r_input.data.shape[0], ("The Variables shape does not allow matrix multiplication: " + str(l_input.data.shape) + " @ " + str(r_input.data.shape))
        
        super(MatMul, self).forward(l_input, r_input)
        
        result = l_input.data @ r_input.data
        if not isinstance(result, FloatTensor): # make sure you have a tensor (not a float/int)
            result = FloatTensor([result])
        return hotgrad.variable.Variable(result, previous_op=self)
        
    def backward(self, grad):
        """ Propagate the gradient to the two input Variables. """
        # first transpose the two input then do the matrix multiplication with the received gradient
        r_input_t = self.r_input.data.t() if self.r_input.data.dim()==2 else self.r_input.data
        l_input_t = self.l_input.data.t() if self.l_input.data.dim()==2 else self.l_input.data
        
        l_grad = grad @ r_input_t
        r_grad = l_input_t @ grad
        # if both an input and grad had shape (1,) then the output is just a float: convert back to FloatTensor
        if not isinstance(l_grad, FloatTensor):
            l_grad = FloatTensor([l_grad])
        if not isinstance(r_grad, FloatTensor):
            r_grad = FloatTensor([r_grad])

        self.l_input.backward(grad = l_grad)
        self.r_input.backward(grad = r_grad)
    
class Pow(hotgrad.module.Module2Operands):        
    def forward(self, l_input, r_input):
        """ Compute the forward pass. """
        if isinstance(r_input, int):
            r_input = FloatTensor([r_input])
        assert (l_input.shape == r_input.shape or r_input.shape == (1,)), "The exponent must have the same shape as the base or must have shape (1,)"
        
        super(Pow, self).forward(l_input, r_input)

        return hotgrad.variable.Variable(l_input.data.pow(r_input), previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient only to the base Variable. """
        # note: r_input (the exponent) is a FloatTensor since we do not compute the gradiend wrt it
        l_grad = grad * (self.r_input * self.l_input.data.pow(self.r_input - 1))
        self.l_input.backward(grad = l_grad)

class Mean(Module1Operand):
    def forward(self, input):
        """ Compute the forward pass. """
        super(Mean, self).forward(input)
        
        return hotgrad.variable.Variable(FloatTensor([input.data.mean()]), previous_op=self)
    
    def backward(self, grad):
        """ Propagate the gradient to the input Variable. """
        if grad.shape != (1,):
            raise BackwardException("The Mean module must receive a gradient of shape (1,)")
        
        num_entries = 1
        for dimension in list(self.input.shape):
            num_entries = num_entries * dimension
        
        # compute the gradient wrt the input
        input_grad = FloatTensor(self.input.shape).fill_(1/num_entries)
        self.input.backward(grad = input_grad*grad)