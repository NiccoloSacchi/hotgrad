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
        return self.loss_criterion.forward(predicted_value, target)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, batch_size=20, epochs=25, verbose=True):
        compute_test_err = X_test is not None and y_test is not None
        
        for e in range(0, epochs):
            sum_loss_train = 0
            for b in range(0, X_train.size(0), batch_size):
                output = self.forward(X_train[b : b+batch_size])
                loss = self.criterion(output, y_train[b : b+batch_size])
                
                sum_loss_train += loss.data[0]
                loss.backward()
                self.optimizer.step()
                
                # zero_grad
                self.zero_grad()
    
    def zero_grad(self):
        for variable in self.variables:
            variable.zero_grad()
    # should call loss.backward()
    def backward(self):
        return 0