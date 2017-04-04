import random

import numpy as np

class Network(object):

	def __init__(self, sizes):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.initialize_weights()

	def initialize_weights(self):
		self.weights = [np.random.randn(y,x)/np.sqrt(x)
				for x,y in zip(self.sizes[:-1], self.sizes[1:])]
		self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
	
	def feedforward(self, x):
		a = x
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w,a) + b)
		return a

	def cost(self, a, y):
		y = vectorize_y(y)
		return -np.sum(np.nan_to_num(y*np.log(a) + (1-y)*np.log(1-a)))

	def total_cost(self, data, lmbda=0):
		cost = 0
		n = len(data)
		for x,y in data:
			x = x.reshape(len(x),1)
			y = y.reshape(len(y),1)
			h = feedforward(x)
			cost += 1.0/n * cost(h,y)
		cost += lmbda/(2*n) * np.sum(np.linalg.norm(w)**2 for w in self.weights)
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
		y = vectorize_y(y)
		delta = a - y
		grad_b[-1] = delta
		grad_w[-1] = np.dot(delta, activations[-2].T)

		for l in range(2,self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].T, delta) * sp
			grad_b[-l] = delta
			grad_w[-l] = np.dot(delta, activations[-l-1].T)
		return (grad_b, grad_w)
	
	def SGD(self, training_data, test_data, epochs, mini_batch_size, eta,
			lmbda = 0):
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k + mini_batch_size]
					for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta, lmbda, n)
			print('Epoch {0}: {1}% accuracy'.format(
				j, 100.0*self.evaluate(test_data)/len(test_data)))
	
		test_accuracy = self.evaluate(test_data)/len(test_data)
		train_accuracy = self.evaluate(training_data)/len(training_data)
		return train_accuracy, test_accuracy

	def update_mini_batch(self, batch, eta, lmbda, n):
		grad_b = [np.zeros(b.shape) for b in list(self.biases)]
		grad_w = [np.zeros(w.shape) for w in list(self.weights)]
		for x, y in batch:
			x = x.reshape(len(x),1)
			y = y.reshape(len(y),1)
			delta_grad_b, delta_grad_w = self.backprop(x, y)
			#print (delta_grad_w[1])
			grad_b = [gb + dgb for gb, dgb in zip(grad_b, delta_grad_b)]
			grad_w = [gw + dgw for gw, dgw in zip(grad_w, delta_grad_w)]
		self.weights = [(1 - eta*lmbda/n)*w - eta/len(batch)*gw
				for w, gw in zip(self.weights, grad_w)]
		self.biases = [b - eta/len(batch)*gb 
				for b, gb in zip(self.biases, grad_b)]
	
	def evaluate(self, test_data):
		correct = 0
		for x,y in test_data:
			x = x.reshape(len(x),1)
			y = y.reshape(len(y),1)
			if (np.argmax(self.feedforward(x)) == y):
				correct += 1
		return correct

def vectorize_y(j):
	e = np.zeros((self.sizes[-1],1))
	e[j] = 1
	return e

def sigmoid(z):
	z = np.clip(z, -500, 500)
	return 1.0/(1.0 + np.exp(-z) )

def sigmoid_prime(z):
	return sigmoid(z) * ( 1 - sigmoid(z))

