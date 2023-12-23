from datastore import Datastore
from ffnn import FFNN
import numpy as np

w1, b1 = np.zeros( (784, 300) ), np.random.multivariate_normal( [0 for i in range(300)], list(np.identity(300)) )
w2, b3 = np.zeros( (300, 150) ), np.random.multivariate_normal( [0 for i in range(150)], list(np.identity(150)) )
w3, b3 = np.zeros( (150, 75) ), np.random.multivariate_normal( [0 for i in range(75)], list(np.identity(75)) )
w4, b4 = np.zeros( (75, 10) ), np.random.multivariate_normal( [0 for i in range(10)], list(np.identity(10)) )
lr = 0.05

ffnn = FFNN()
ffnn.set_weights([w1, w2, w3, w4])
ffnn.set_biases([b1, b2, b3, b4])

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def relu(x):
    return np.max(x, 0)

def forward(x):
    act1 = sigmoid(w1.T @ x + b1)
    act2 = sigmoid(w2.T @ act1 + b2)
    act3 = sigmoid(w3.T @ act2 + b3)
    act4 = sigmoid(w4.T @ act3 + b4)
    ffnn.set_activations([act1, act2, act3, act4])
    return act4

def man_backward(target, optim='gradient-descent'):
    activations, weights, biases = ffnn.activations(), ffnn.weights(), ffnn.biases()
    output_grad =
    activation_gradients = [-np.sum(target.T @ (1/activations[4]))]
    unactivated_gradients = []
    weight_gradients = [activation_gradients[-1] * (activations[3] @ (activations[4]*(1 - activations[4])).T )]

    for i in range(1, len(weights)):


def train(datastore):
    train_examples, split_examples, test_examples = datastore.feed_examples()



print(w1, b1)
