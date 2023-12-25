from ffnn import FFNN
from utils import cross_entropy_loss
import numpy as np
import datasets
from tqdm import tqdm
from PIL import Image
import sys
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
debug_state = int(sys.argv[1])

mnist = datasets.load_dataset("mnist")
train_ds = list( map(np.array, mnist['train'][:]['image']) )
train_label = mnist['train'][:]['label']
test_ds =  list( map(np.array, mnist['test'][:]['image']) )
test_label = mnist['test'][:]['label']
input_model = FFNN()

def train(input_model: FFNN, full_train = []):
    example = np.ones( (784, 1) )

    for epoch in tqdm(range(10)):
        loss = 0

        weight_gradients = []
        bias_gradients = []

        for idx, (image, label) in (enumerate(full_train[0:100])):
            if idx > 0 and idx % 10 == 0:
                logging.warning("Average epoch loss is {}".format(loss/float(idx)))

            image = np.reshape(image, (784, 1) )
            output = input_model.step_forward(image)
            logging.info("output: {}".format(output))
            target = np.zeros( (10, 1) )
            target[label, 0] = 1

            wt_grad, bias_grad = input_model.step_backward(target)
            if ( len(weight_gradients) == 0):
                weight_gradients = wt_grad
                bias_gradients = bias_grad
            else:
                weight_gradients = [weight_gradients[i] + wt_grad[i] for i in range(len(weight_gradients))]
                bias_gradients = [bias_gradients[i] + bias_grad[i] for i in range(len(bias_gradients))]
            loss += cross_entropy_loss(target, output)

        weight_gradients = [weight_gradients[i]/float(100.0) for i in range(len(weight_gradients))]
        bias_gradients = [bias_gradients[i]/float(100.0) for i in range(len(bias_gradients)) ]

        logging.warning("Average epoch loss is {}".format(loss/len(full_train)))
        
        #Perform batch gradient descent

        logging.warning("First weight gradient: {}".format(weight_gradients[0]))

        for i in range(len(weight_gradients)):
            input_model.weights[i] = input_model.weights[i] - (input_model.lr * weight_gradients[i])
            input_model.biases[i] = input_model.biases[i] - (input_model.lr * bias_gradients[i])

    logging.warning("Finished training!")
    logging.warning("Weight dims: {}".format( list(map(lambda x : x.shape, input_model.weights))) )
    logging.warning("Weights: {}".format( input_model.weights) )
    logging.warning("Biases: {}".format( input_model.biases) )
    return input_model

def test( trained_model: FFNN, full_test = []) -> float:
    accuracy = 0
    for idx, (image, label) in enumerate(full_test):
        output = trained_model.step_forward(np.reshape(image, (784, 1) ) )
        print("output: {}".format(output))
        mle = np.argmax(output)
        print("mle: {}, label: {}".format(mle, label))
        accuracy += (1 if mle == label else 0)

    result = accuracy/float(len(full_test))
    print ("Accuracy is {}".format( result ) )
    return result


if debug_state == 0:
    logging.warning("RUNNING DEBUG STATE ZERO")
    
    full_train = list(zip(train_ds, train_label))
    trained_model = train(input_model, full_train)
    full_test = list(zip(test_ds, test_label))
    #test(trained_model, full_test)
    print("Output 1: {}".format(trained_model.step_forward( np.reshape(full_test[0][0], (784, 1) ) )) )
    print("Label 1: {}".format(full_test[0][1]))
    print("Output 2: {}".format(trained_model.step_forward( np.reshape(full_test[1][0], (784, 1) )) ) )
    print("Label 2: {}".format(full_test[1][1]))

elif debug_state == 1:
    logging.warning("RUNNING DEBUG STATE ONE")
    print(train_ds[10])
    example = train_ds[10].reshape( (784, 1) )
    label = np.zeros( (10, 1) )
    label[train_label[10], 0] = 1

    for _ in range(20):
        output = input_model.step_forward(example)
        logging.warning("Categorical cross entropy error: {}".format(cross_entropy_loss(label, output)))
        wt_grad, bias_grad = input_model.step_backward(label, 'stochastic-gradient-descent')

elif debug_state == 2:
    logging.warning("RUNNING DEBUG STATE TWO")

    example = train_ds[0].reshape( (784, 1) )
    logging.info("Label: {}".format(train_label[0]))
    output = input_model.step_forward(example)
    logging.info("Output: {}".format(output))
    target = np.zeros( (10, 1) )
    target[0, 0] = 1
    logging.info("weights before: {}".format(input_model.weights))
    input_model.step_backward(target)
    logging.info("weights after: {}".format(input_model.weights))
    logging.info("weight dimensions: {}", list(map(lambda x : x.shape, input_model.weights)) ) 
