from utils import sigmoid, softmax, relu
from typing import List
import numpy as np
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class FFNN:
    def __init__(self):
        w1, b1 = np.random.rand(784, 300), np.expand_dims(np.random.multivariate_normal( [0 for i in range(300)], list(np.identity(300)) ), axis=1)
        w2, b2 = np.random.rand(300, 150), np.expand_dims(np.random.multivariate_normal( [0 for i in range(150)], list(np.identity(150)) ), axis=1)
        w3, b3 = np.random.rand(150, 75), np.expand_dims(np.random.multivariate_normal( [0 for i in range(75)], list(np.identity(75)) ), axis=1)
        w4, b4 = np.random.rand(75, 10), np.expand_dims(np.random.multivariate_normal( [0 for i in range(10)], list(np.identity(10)) ), axis=1)
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
        #print("x shape: {}".format(x.shape))
        act1 = sigmoid(w1.T @ x + b1)
        #print("act1 shape: {}".format(act1.shape))
        act2 = sigmoid(w2.T @ act1 + b2)
        #print("act2 shape: {}".format(act2.shape))
        act3 = sigmoid(w3.T @ act2 + b3)
        #print("act3 shape: {}".format(act3.shape))
        act4 = softmax(w4.T @ act3 + b4)
        #print("act4 shape: {}".format(act4.shape))
        self.activations = [x, act1, act2, act3, act4]

  #     print(list(map(lambda x : x.shape, self.activations)))
        return act4

    def step_backward(self, target, optim='batch-gradient-descent'):
        output_grad = -target / self.activations[-1]  # gradient for cross entropy loss
        #print("upd_activations shape: {}".format(upd_activations.shape))
        # assume that every layer uses sigmoid activation function
        activation_gradients = [output_grad]
        # here, compute dL/dz^n = f'(z_n) * (dL/da^n), where f is softmax
        #print ( "sigmoid gradient shape: {}".format( ((activations[-1] @ (1 - activations[-1].T) )).shape ) )
        sigmoid_jacobian = np.identity(self.activations[-1].shape[0]) * ( (1 - self.activations[-1]) * self.activations[-1] )
        unactivated_gradients = [ sigmoid_jacobian @ output_grad]
        weight_gradients = [ self.activations[-2] @ (unactivated_gradients[-1]).T ]
        bias_gradients = unactivated_gradients

        for i in range(len(self.activations)-2, 0, -1):
            # note here: the gradient of sigmoid(x) = (sigmoid(x)) * (1 - sigmoid(x))
            activation_gradients.append(  ( self.weights[i]) @ (unactivated_gradients[-1]) )
            sigmoid_jacobian = np.identity(self.activations[i].shape[0]) * ( (1 - self.activations[i]) * self.activations[i] )
            unactivated_gradients.append( sigmoid_jacobian @ (activation_gradients[-1]) )
            weight_gradients.append( self.activations[i-1] @ (unactivated_gradients[-1]).T )
        
        if ( optim == 'stochastic-gradient-descent'):
            weight_gradients.reverse()
            bias_gradients.reverse()

            for i in range(len(weight_gradients)):
                self.weights[i] = self.weights[i] - (self.lr * weight_gradients[i])
                self.biases[i] = self.biases[i] - (self.lr * bias_gradients[i])
        else:       
            weight_gradients.reverse()
            bias_gradients.reverse()
        
        return weight_gradients, bias_gradients
