from torch import FloatTensor
import numpy as np

def generate_dataset(N = 1000):
    r = 1/np.sqrt(2*np.pi)
    center_x = 0.5
    center_y = 0.5
    X_train = FloatTensor(N, 2).uniform_(0, 1)
    X_test  =  FloatTensor(N, 2).uniform_(0, 1)
    y_train = (((X_train[:, 0]-center_x)**2 + (X_train[:, 1]-center_y)**2) > r**2).type(FloatTensor)
    y_test  = (((X_test[:, 0]-center_x)**2 + (X_test[:, 1]-center_y)**2) > r**2).type(FloatTensor)
    return X_train, X_test, y_train, y_test
