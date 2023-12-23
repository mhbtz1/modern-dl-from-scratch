from utils import sigmoid, softmax, relu
from typing import List
import numpy as np

class FFNN:
    def __init__(self):
        w1, b1 = np.zeros( (784, 300) ), np.expand_dims(np.random.multivariate_normal( [0 for i in range(300)], list(np.identity(300)) ), axis=1)
        w2, b2 = np.zeros( (300, 150) ), np.expand_dims(np.random.multivariate_normal( [0 for i in range(150)], list(np.identity(150)) ), axis=1)
        w3, b3 = np.zeros( (150, 75) ), np.expand_dims(np.random.multivariate_normal( [0 for i in range(75)], list(np.identity(75)) ), axis=1)
        w4, b4 = np.zeros( (75, 10) ), np.expand_dims(np.random.multivariate_normal( [0 for i in range(10)], list(np.identity(10)) ), axis=1)
        lr = 0.05

        self.weights : List[np.ndarray] = [w1, w2, w3, w4]
        self.biases : List[np.ndarray] = [b1, b2, b3, b4]
        self.activations : List[np.array] = []
        self.lr : np.float64 = lr

    @property
    def _activations(self):
        return self.activations

    @property
    def _weights(self):
        return self.weights

    @property
    def _biases(self):
        return self.biases

    @_weights.setter
    def set_weights(self, nweights):
        self.weights = nweights

    @_activations.setter
    def set_activations(self, nactive):
        self.activations = nactive

    @_biases.setter
    def set_biases(self, nbiases):
        self.biases = nbiases

    def step_forward(self, x):
        w1, w2, w3, w4 = self.weights
        b1, b2, b3, b4 = self.biases
        act1 = sigmoid(w1.T @ x + b1)
        act2 = sigmoid(w2.T @ act1 + b2)
        act3 = sigmoid(w3.T @ act2 + b3)
        act4 = softmax(w4.T @ act3 + b4)
        self.activations = [x, act1, act2, act3, act4]

        print(list(map(lambda x : x.shape, self.activations)))
        return act4

    def step_backward(self, target, optim='gradient-descent'):
        activations, weights, biases = self.activations, self.weights, self.biases

        output_grad = np.zeros(target.shape)
        output_grad[np.argmax(target,axis=1), 0] = -1 / activations[-1][np.argmax(target)] # gradient for cross entropy loss

        # assume that every layer uses sigmoid activation function

        activation_gradients = [output_grad]

        # here, compute dL/dz^n = f'(z_n) * (dL/da^n), where f is softmax
        unactivated_gradients = [ ((activations[-1] @ (1 - activations[-1].T) )) @ output_grad ]
        weight_gradients = [ activations[-2] @ (unactivated_gradients[-1]).T ]
        bias_gradients = unactivated_gradients

        for i in range(len(activations)-2, 0, -1):
            # note here: the gradient of sigmoid(x) = (sigmoid(x)) * (1 - sigmoid(x))

            #print( "lmul activation gradient shape: {}".format(weights[i].shape) )
            #print ( "rmul activation gradient shape: {}".format(unactivated_gradients[-1].shape) )
            activation_gradients.append(  ( weights[i]) @ unactivated_gradients[-1] )

            #print( "lmul shape: {}".format( (activations[i] @ ( 1 - activations[i].T)).shape ) )
            #print ( "rmul shape: {}".format( activation_gradients[-1].shape ) )
            unactivated_gradients.append( (activations[i] @ ( 1 - activations[i].T)) @ activation_gradients[-1] )

            #print( "lmul shape: {}".format( (activations[i-1].shape) ) )
            #print ( "rmul shape: {}".format( unactivated_gradients[-1].T.shape ) )
            weight_gradients.append( activations[i-1] @ (unactivated_gradients[-1]).T )

        print( list(map(lambda x: x.shape, weight_gradients)) )
        print( list(map(lambda x : x.shape, activation_gradients)) )
        print( list(map(lambda x : x.shape, unactivated_gradients)) )


        weight_gradients.reverse()
        bias_gradients.reverse()
        for i in range(len(weight_gradients)):
            weights[i] = weights[i] - (self.lr * weight_gradients[i])
            biases[i] = biases[i] - (self.lr * bias_gradients[i])


