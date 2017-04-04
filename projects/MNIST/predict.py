import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../../')
from DataSets.MNIST import mnist_reader
from projects.Network import Network

reader = mnist_reader.mnist_reader('../../DataSets/MNIST/')
training_data, test_data = reader.load()
training_data = list(training_data)
test_data = list(test_data)


np.seterr(all='raise')
net = Network([784, 200, 10])

train_accuracies = []
test_accuracies = []

train, test = net.SGD(training_data, test_data, 20, 10, 0.3 , 5)
