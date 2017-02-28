from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

def rbf_kernel_pca(X, gamma, n_components):
	"""
	RBF kernel PCA implementation.

	Parameters
	-----------
	X: {NumPy ndarray}, shape = [n_samples, n_features]

	gamma: float
		Tuning parameter of teh RBF kernel

	n_components: int
		Number of principal components to return

	Returns
	---------
	X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
		projected dataset

	lambdas: list
		Eigenvalues

	"""
	# calculate pairwise squared Euclidean distances
	# in the MxN dimensional dataset.
	sq_dists = pdist(X, 'sqeuclidean')

	# convert pairwise distances into a square matrix
	mat_sq_dists = squareform(sq_dists)

	# Compute the symmetric kernel matrix
	K = exp(-gamma * mat_sq_dists)

	#Center the kernel matrix
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) 

	# obtaining eigenpairs from the centered kernel matrix
	# numpy.eigh returns them in sorted order
	eigvals, eigvecs = eigh(K)

	# Collect the top k eigenvectors (projected samples)
	alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
	
	# Collect the corresponding eigenvalues
	lambdas = [eigvals[-i] for i in range(1, n_components+1)]

	return alphas, lambdas

# Example 1 -- Seperating half-moon shapes
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=0)

# plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

# First attempt -- Use linear PCA
from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
# ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)

# ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)

# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1,1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()

# Second attempt -- Use rbf_kernel_pca
from matplotlib.ticker import FormatStrFormatter
#X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
# ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)

# ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)

# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1,1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# plt.show()


# Example 2 -- separating concentric circles
from sklearn.datasets import make_circles
X, y =make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
# plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)

# X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
# ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)

# ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y==1, 0], np.zeros((5030,1))-0.02, color='blue', marker='o', alpha=0.5)

# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# ax[1].set_ylim([-1,1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# plt.show()

# Using scikit-learn's kernel_principal_component
from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()





