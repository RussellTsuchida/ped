import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lib.plotters import matplotlib_config
from model import PrincipalEquilibriumDimensions

BATCH_SIZE_TRAIN    = 100
NUM_EPOCHS          = 100
LAMBDA              = 0.1
DIST                = 'bernoulli'
BASIS               = 'non-ortho'
BIG_DIM             = 10
LOW_DIM             = 2
NUM_POINTS          = 1000
OUTPUT_DIR          = 'outputs/synthetic/'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
matplotlib_config()

def generate_data(num_points, batch_size, big_dim, low_dim=2, num_clusters=9):
    # Generate ground truth data from a model that matches our setup exactly
    assert low_dim ==2

    # Generate a basis 
    W = np.random.normal(0, 1, (big_dim, low_dim))

    # generate some points 
    grid_size = np.sqrt(num_clusters)
    assert np.abs(int(grid_size) - grid_size) < 0.4
    x = np.arange(-int(np.floor(grid_size/2)), int(grid_size-np.floor(grid_size/2)))
    z_means = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])

    z = np.random.normal(z_means, 0.2*np.ones_like(z_means), ( num_points, num_clusters, low_dim))
    labels = np.zeros((num_points, num_clusters))
    for c in range(num_clusters):
        labels[:,c] = c
    z = z.reshape((-1, low_dim)).T
    labels = torch.Tensor(labels.reshape((-1,)))

    eta = (W @ z).T # size = (num_points*num_clusters, big_dim)
    bias = np.random.normal(0, 0.1, (1, big_dim))
    eta = eta + bias

    p = np.exp(eta)/(1+np.exp(eta))
    Y = torch.Tensor(np.random.binomial(1, p))

    data = torch.utils.data.TensorDataset(Y, labels)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size)

    return data_loader, data, Y, labels

data_loader, data_set, true_Y, true_labels = generate_data(NUM_POINTS, BATCH_SIZE_TRAIN, BIG_DIM, LOW_DIM)


############################################### Fit PED then visualise
ped = PrincipalEquilibriumDimensions(2)

new_x, labels = ped.fit_transform(data_loader, lamb=LAMBDA, basis=BASIS,
    dist=DIST, weight_decay=2, num_epochs=NUM_EPOCHS, plot_bool=False)

#new_x = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(new_x)
plt.scatter(new_x[:,0], new_x[:,1], c=labels, s=1)
plt.savefig(OUTPUT_DIR + 'ped.png')
plt.close()

#################################################### Fit PCA then visualise
scaler = StandardScaler()
data_scaled = scaler.fit_transform(true_Y.numpy())
new_x = PCA(n_components=2).fit_transform(data_scaled)
#new_x = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(new_x)
plt.scatter(new_x[:,0], new_x[:,1], c=true_labels.numpy(), s=1)
plt.savefig(OUTPUT_DIR + 'pca.png')
plt.close()
