import numpy as np
from typing import Tuple, Callable

def sigmoid(x: np.array) -> np.array:
    return 1/(1 + np.exp(-x))
def relu(x : np.array) -> np.array:
    return np.max(x, 0)
def softmax(x : np.array) -> np.array:
    return (np.exp(x)/(np.sum(np.exp(x), axis=0)))

def cross_entropy_loss(target : np.array, actual: np.array) -> np.float64:
    return -target.T @ np.log(actual)

class Sigmoid():
    def __init__(self):
        pass

    def __call__(self, x):
        return 1/(1 + np.exp(-x))

    def __grad__(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

'''
def sigmoid(x : np.array) -> Tuple[np.array, Callable[ [np.ndarray], np.ndarray ]]:
    raw_sigmoid = lambda x : 1/(1 + np.exp(-x))
    return (  raw_sigmoid(x), lambda x : raw_sigmoid(x) * (1 - raw_sigmoid(x).T) )
def relu(x : np.array) -> Tuple[np.array, Callable [ [np.ndarray], np.ndarray ]]:
    return ( np.max(x, 0), np.apply_along_axis(lambda x: 1 if x > 0 else 0, 0, x) )
def softmax(x : np.array) -> Tuple[np.array, Callable[ [np.ndarray], np.ndarray ]]:
    return( np.exp(x)/(np.sum(np.exp(x)))
'''
