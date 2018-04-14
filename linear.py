from module import Module
from variable import Variable
from torch import Tensor

class Linear(Module):
    """
    Implements a fully connected layer
    """
    
    def __init__(self, input_features, output_features, bias=None):
        self.input_features = input_features
        self.output_features = output_features
        #self.bias = bias
        self.weight = Variable(input_features,output_features)
#         self.req_grad = True
    
    def forward(self, input):
#         return 0
        # input: N x input_features matrix
        # weigth: input_features x output_features matrix
        # returns N x output_features matrix
        return input.mm(Tensor(self.weight))
        
    def backward(self, input):
        return 0