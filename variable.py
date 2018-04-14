from torch import Tensor

class Variable(Tensor):
    """
    A wrapper for Tensor that stores the gradient
    """
    def __init__(self, *args):
        self.gradient = Tensor(self.size()).zero_()

    def asd(self):
        return "asdlol"