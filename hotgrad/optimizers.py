# -*- coding: utf-8 -*-

""" Implementation of the optimizers. """

class SGD():
    def __init__(self, lr = 0.01):
        self.lr = lr
        
    def set_params(self, params):
        self.params = params
        
    def params(self):
        return self.params
        
    """ updates all the inputs of the modules with their gradient """
    def step(self):
        for param in self.params():
#            param.data = param.data - self.lr * param.grad
            param.data.sub_(self.lr * param.grad)