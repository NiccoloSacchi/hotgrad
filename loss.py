import module
from module import Module

class LossMSE(Module):
    """
    Computes the Mean Squared Error:
        formula
    """
    
    def __init__(self):
        self.req_grad = False
    
    def forward(self, input, target):
        return (input - target).pow(2).mean(0)