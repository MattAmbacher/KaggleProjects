import os
import struct

import numpy as np

class mnist_reader(object):

	def __init__(self, path):
		self.trimg = os.path.join(path, 'train-images-idx3-ubyte')
		self.trlbl = os.path.join(path, 'train-labels-idx1-ubyte')
		self.teimg = os.path.join(path, 't10k-images-idx3-ubyte')
		self.telbl = os.path.join(path, 't10k-labels-idx1-ubyte')
	
	def read_label(self, fname):
		with open(fname, 'rb') as f:
			magic, num = struct.unpack('>II', f.read(8))
			labels = np.fromfile(fname, dtype=np.uint8)[8:]
			labels = labels.reshape(num,1)
		return labels

	def read_images(self, fname):
		with open(fname, 'rb') as f:
			magic, num, rows, cols = struct.unpack('>IIII', f.read(2**4))
			imgs = np.fromfile(fname, dtype=np.uint8)[2**4:]
			imgs = imgs.reshape(num, rows*cols)
		return imgs
	
	def pack(self, imgs, labels):
		return zip(imgs, labels)

	
	def load(self):
		training_labels = self.read_label(self.trlbl)
		training_images = self.read_images(self.trimg)
		test_labels = self.read_label(self.telbl)
		test_images = self.read_images(self.teimg)

		training_data = self.pack(training_images, training_labels)
		test_data = self.pack(test_images, test_labels)
		return (training_data, test_data)
