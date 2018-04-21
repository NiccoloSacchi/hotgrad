from torch import FloatTensor
import numpy as np
from hotgrad.variable import Variable

def generate_dataset_aux(N, one_hot_encoding):
    """ Generate a training and a test set of N points of 2 dimensions, each sampled uniformly in [0, 1]. The
    label is 0 outside the disk of radius 1/sqrt(2π) and 1 inside. """
    
    r = 1/np.sqrt(2*np.pi)
    center_x = 0.5
    center_y = 0.5
    X_train = FloatTensor(N, 2).uniform_(0, 1)
    X_test  = FloatTensor(N, 2).uniform_(0, 1)
    
    y_train = (((X_train[:, 0]-center_x)**2 + (X_train[:, 1]-center_y)**2) > r**2).type(FloatTensor)
    y_test  = (((X_test[:, 0]-center_x)**2 + (X_test[:, 1]-center_y)**2) > r**2).type(FloatTensor)
    
    if one_hot_encoding:
        y_train_hot = FloatTensor(y_train.shape[0], 2)
        y_train_hot[:, 0] = y_train==0
        y_train_hot[:, 1] = y_train==1
        
        y_test_hot = FloatTensor(y_test.shape[0], 2)
        y_test_hot[:, 0] = y_test==0
        y_test_hot[:, 1] = y_test==1
        return X_train, X_test, y_train_hot, y_test_hot
    
    return X_train, X_test, y_train, y_test

def generate_dataset(N = 1000, one_hot_encoding=False):
    X_train, X_test, y_train, y_test =  generate_dataset_aux(N, one_hot_encoding)
    return Variable(X_train), Variable(X_test), Variable(y_test), Variable(y_test)