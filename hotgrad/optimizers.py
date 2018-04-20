# -*- coding: utf-8 -*-

""" Implementation of the optimizers. """

class SGD():
    def __init__(self, modules, lr = 0.01):
        self.modules = modules
        self.lr = lr
        
    """ updates all the inputs of the modules with their gradient """
    def step(self):
        for module in self.modules:
            for param in module.params():
                param.data = param.data - self.lr * param.grad