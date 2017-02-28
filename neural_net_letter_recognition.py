import os
import struct
import numpy as np 
from NeuralNet import *

def load_mnist(path, kind='train'):
	"""Load MNIST data from 'path'"""
	labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
	images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels  = np.fromfile(lbpath, dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols= struct.unpack(">IIII", imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

	return images, labels

X_train, y_train = load_mnist(r"C:\Users\Owen\Desktop\Python Machine Learning\Chapter 12\mnist", kind='train')
print(y_train.shape[0])

#print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

#print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))


import matplotlib.pyplot as plt 
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
	img = X_train[y_train == 7][i].reshape(28, 28)
	ax[i].imshow(img, cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
# np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
# np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
# np.savetxt('test_labels.csv', y_train, fmt='%i', delimiter=',')

"""Do this to load the data from the csv files"""
# X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
# y_train = np.genfromtxt('train_labels.csv', dtype=int, delimter=',')
# X_test = np.genfromtxt('test_img.csv', dtype=int, delimter=',')
# y_test = np.genfromtxt('test_labels.csv', dtype=int, delimter=',')

import matplotlib.pyplot as plt 

nn = NeuralNetMLP(n_output=10,
				  n_features=X_train.shape[1],
				  n_hidden=50,
				  l2=0.1,
				  l1=0.0,
				  epochs=1000,
				  eta=0.001,
				  alpha=0.001,
				  decrease_const=0.00001,
				  shuffle=True,
				  minibatches=50,
				  random_state=1)

nn.fit(X_train, y_train, print_progress=True)

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Testing accuracy: %.2f%%' % (acc * 100))









