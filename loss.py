from module import Module

class LossMSE(Module):
    """
    Computes the Mean Squared Error:
        formula
    """
    
    # TODO: define lossMSE function and its derivative and generalize 
    # this approach for other loss functions as well
    def forward(self, input, target):
        self.input = input
        self.target = target
        return (target - input).pow(2).mean(0)
    
    # TODO: generalize for all the loss functions
    def MSE_derivative(self, input, target):
        return 2*(input - target).mean(0)
    
    def backward(self):
        # can be either dl_dx or dl_ds, depending on whether
        # an activation function is put after the last fully connected layer
        dl_dx = self.MSE_derivative(self.input, self.target)
        