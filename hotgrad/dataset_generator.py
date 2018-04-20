import random
import math
from torch import FloatTensor

def generate_dataset():
    N = 1000
    r = 1/math.sqrt(2*math.pi)
    X_train = FloatTensor([[random.uniform(0, 1), random.uniform(0, 1)] for a in range(N)])
    X_test  = FloatTensor([[random.uniform(0, 1), random.uniform(0, 1)] for a in range(N)])
    y_train = (X_train[:, 0]**2 + X_train[:, 1]**2 > r**2).type(FloatTensor)
    y_test  = (X_test[:, 0]**2 + X_test[:, 1]**2 > r**2).type(FloatTensor)
    return X_train, X_test, y_train, y_test
