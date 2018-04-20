# -*- coding: utf-8 -*-
""" The Variable object stores the previous operation so to allow the gradient
to packpropagate to the previous operation. """

from torch import is_tensor, FloatTensor

from hotgrad.module import Module
from hotgrad.functions.operands import Mul, Mean
from hotgrad.exceptions import NotImplicitGradient

# TODO define if we want to handle broadcasting in the gradient computation (probabily not hard)
# for now assume all passed parameters are of class Variable (except for the pow())
# TODO using assert in this way is not the best practice
class Variable():
    """ Variable are the basic """
    def __init__(self, data, previous_op=None, requires_grad=False):
        assert(is_tensor(data))
        assert(isinstance(previous_op, Module) or previous_op is None)
        assert(isinstance(requires_grad, bool))
        
        self.data = data
        self.shape = data.shape
        self.previous_op = previous_op
        self.requires_grad = requires_grad
        self.grad = FloatTensor(data.shape).zero_()

    def __mul__(self, other):
        """ Multiplies this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        return Mul(self, other)()
    
    def __truediv__(self, other):
        """ Divides this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        print("div")
        return 
    
    def __sub__(self, other):
        """ Subtracts this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        print("sub")
        return 
    
    def __add__(self, other):
        """ Add this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        print("add")
        return 
    
    def __pow__(self, other):
        """ Compute the power of this Variable (element-wise) by a constant, 
        i.e. 'other' can only be of type int/float."""
        print("pow")
        return
    
    def __matmul__(self, other):
        """ Multiplies this Variable by another Variable, i.e. 'other' can only 
        be of type Variable and its shape has to allow for matric multiplication."""
        print("matmul")
        return
    
    def mean(self):
        return Mean(self).forward()
    
    def pow(self, other):
        return self.pow(other)
    
    def backward(self, grad=None):
        # if the backpropagation starts here then shape of this Variable must be (1,)
        # (the gradient can be computed implicitly only for scalar output)
        if grad is None: 
            # the backward propagation starts here
            if self.data.shape != (1,):
                # Cannot compute implicitly the gradient
                raise NotImplicitGradient()
            else:
                # Can compute implicitly the gradient => set the current gradient to 1 
                # (derivative of this variable with respec this variable)
                grad = FloatTensor([1]) # TODO set in the function call
        
        # check if this variable requires the gradient. If so then update it's local gradient.
        if (self.requires_grad is not None and grad is not None):
            assert(is_tensor(grad))
            assert(grad.shape == self.data.shape)
            self.grad += grad
            
        # finally propagate the gradient
        if (self.previous_op is not None):
            self.previous_op.backward(grad) # propagate the gradient
            
    def __str__(self):
        return "Variable containing:" + str(self.data)
    
    def __repr__(self):
        return self.__str__()
