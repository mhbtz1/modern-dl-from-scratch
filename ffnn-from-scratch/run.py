from ffnn import FFNN
import cross_entropy_loss from utils
import numpy as np
import datasets
import tqdm
from PIL import Image

mnist = datasets.load_dataset("mnist")
train_ds = list( map(np.array, mnist['train'][:]['image']) )
train_label = mnist['train'][:]['label']
test_ds =  list( map(np.array, mnist['test'][:]['image']) )
test_label = mnist['test'][:]['label']
input_model = FFNN()

def train(input_model: FFNN, full_train = []):
    example = np.ones( (784, 1) )

    for epoch in range(10):
        loss = 0
        for idx, (image, label) in tqdm(enumerate(full_train)):
            image = np.reshape(image, (784, 1) )
            output = ffnn.step_forward(image)
            target = np.zeros( (10, 1) )
            target[label, 0] = 1
            ffnn.step_backward(target)
            loss += cross_entropy_loss(target, output)
        print("Average epoch loss is {.2f}".format(loss))
    print("Finished training!")
    return ffnn

def test( input_model: FFNN, full_test = []):



train(input_model, train_ds)
test(input_model, test_ds)

