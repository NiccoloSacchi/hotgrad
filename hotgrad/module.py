# -*- coding: utf-8 -*-
""" Implementation of the "abstract" class representing the module."""

# import torch
# from .variable import Variable

class Module(object):
    """ Base class that provides the interface that all the modules should implement. Each module
    is part of the newtwork and, other that implementing the operation on the input (forward pass),
    has to implement the backward step so to propagate the gradient to the previous modules in the
    network.
    """

    def __call__(self):
        return self.forward()
    
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def requires_gradient(self):
        return self.req_grad
    
    
class Module2Operands(Module):
    """ Base class for modules accepting 2 operands as input. """
    def __init__(self, l_input=None, r_input=None):
#         assert(isinstance(l_input, Variable))
#         assert(isinstance(r_input, Variable))
        
        self.l_input = l_input
        self.r_input = r_input
        return
        
    def __str__(self):
        return (self.__class__.__name__ + " operator:\n" + 
                "\n------ Left operand ------\n" + str(self.l_input) + "--------------------------\n" +
                "\n------ Right operand ------\n" + str(self.r_input) + "--------------------------")
    
    def __repr__(self):
        return self.__str__()
    
    def forward(self):
        """ Applies the operation to the two operands passed during initialization. """
        raise NotImplementedError

        
class Module1Operand(Module):
    """ Base class for modules accepting only one operand as input. """
    def __init__(self, input):
#         assert(isinstance(input, Variable))
        self.input = input
        return
        
    def __str__(self):
        return (self.__class__.__name__ + " operator:\n" + 
                "\n------ Operand ------\n" + str(self.input) + "--------------------------\n")
    
    def __repr__(self):
        return self.__str__()

    def __call__(self):
        return self.forward()
    
    def forward(self):
        """ Computes the forwards pass using the operand passed during initialization. """
        raise NotImplementedError
