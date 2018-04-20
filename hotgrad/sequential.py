# -*- coding: utf-8 -*-

""" Implementation of the Sequential module """

class Sequential(Module):
    """
        modules: list of modules that compose the network
        loss_criterion: the function that is used for computing the loss
        optimizer: the optimized used for updating the gradients
    """
    def __init__(self, modules, loss_criterion, optimizer):
        self.modules = modules
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        
    """
        computes the forward pass of all the modules
    """
    def forward(self, input):
        modules = self.modules.copy()
        first_module = modules.pop()
        module_result = first_module.forward(input)
        
        for module in self.modules:
            module_result = module.forward(module_result)
            
        return module_result
    
    def get_loss(self, predicted_value, target):
        return self.loss_criterion(predicted_value, target)
    
    # should call loss.backward()
    def backward(self):
        return 0