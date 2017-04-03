import sys
import numpy as np

sys.path.append('../../')
from DataSets.MNIST import mnist_reader
from projects.Network import Network

reader = mnist_reader.mnist_reader('../../DataSets/MNIST/')
training_data, test_data = reader.load()

net = Network([748, 25, 10])

net.SGD(training_data, 100, 20, 1)
