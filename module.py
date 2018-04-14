import torch

class Module(object):
    """ Base class that provides the interface that all the modules should implement

    """

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def requires_gradient(self):
        return self.req_grad