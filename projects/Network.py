import random

import numpy as np

class Network(object):

	def __init__(self, sizes):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.weights = self.initialize_weights()

	def initialize_weights(self):
		self.weights = [np.random.randn(y,x)/np.sqrt(x)
				for x,y in zip(self.sizes[:-1], self.sizes[1:])]
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
	
	def feedforward(self, x):
		a = x
		for w, b in self.weights, self.biases:
			a = sigmoid(np.dot(w,a) + b)
		return a
	def cost(self, a, y):
		return -np.sum(np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)))

	def total_cost(self, data, lmbda=0):
		cost = 0
		m = len(data)
		for x,y in data:
			x = x.reshape(len(x),1)
			y = y.reshape(len(y),1)
			h = feedforward(x)
			cost += 1.0/m * cost(h,y)
		cost += lmbda/(2*m) * np.sum(np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def backprop(self, x, y):
		grad_w = [np.zeros(w.shape) for w in self.weights]
		grad_b = [np.zeros(b.shape) for b in self.biases]

		a = x
		activations = [a]
		zs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w,a) + b
			zs.append(z)
			a = sigmoid(np.dot(w,a) + b)
			activations.append(a)
		#backwards through NN
		delta = a - y
		grad_b[-1] = delta
		grad_w[-1] = np.dot(delta, activations[-2].T)

		for l in xrange(2,self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].T, delta) * sp
			grad_b[-l] = delta
			grad_w[-l] = np.dot(delta, activations[-l-1].T)
		return (grad_b, grad_w)
	
	def SGD(self, training_data, epochs, mini_batch_size, eta,
			lmbda = 0):
		m = len(training_data[0])
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k + mini_batch_size]
					for k in range(0, m, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta, lmbda)
			print('Epioch {0} complete'.format(j))
	
	def update_mini_batch(self, batch, eta, lmbda):
		grad_b = [np.zeros(b.shape) for b in self.biases]
		grad_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in batch:
			x = x.reshape(len(x),1)
			y = y.reshape(len(y),1)
			delta_grad_b, delta_grad_w = backprop(x, y)
			grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
			grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]
		self.weights = [w - (eta/len(mini_batch))*gw
				for w, gw in zip(self.weights, grad_w)]
		self.bises = [b - (eta/len(mini_batch))*gb 
				for b, gb, in zip(self.biases, grad_b)]

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * ( 1 - sigmoid(z))
