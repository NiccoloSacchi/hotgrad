# -*- coding: utf-8 -*-
""" Definition of the exceptions. """

class BackwardException(Exception):
    pass

class NotImplicitGradient(Exception):
    def __init__(self):
        super(NotImplicitGradient, self).__init__("The gradient can be computed implicitly only for a scalar output.")
        