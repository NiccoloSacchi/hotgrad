from module import Module
from torch import Tensor

class Relu(Module):
    """
    Implements a fully connected layer
    """
    
    def forward(self, input):
        self.input = input
        res = input.clone()
        res[res<0] = 0
        return res
    
    # TODO: define an activation interface that is implemented by loss and relu?
    def relu_gradient(self, input):
        res = Tensor(input.size()).fill_(1)
        res[res<0] = 0
        return res
        
    # TODO: need to interact with the optimizer
    def backward(self, weights_next_layer, dl_ds_next_layer):
        # dl_ds_next_layer: N x output_features matrix
        # weights_next_layer: input_features x output_features matrix
        # dl_dx: N x input_features matrix
        self.dl_dx = dl_ds_next_layer.mm(weights_next_layer.t())
        self.dl_ds = self.dl_dx * self.relu_gradient(self.input)
    