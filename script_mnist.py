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
LAMBDA              = 1.0
DIST                = 'relu'
BASIS               = 'non-ortho'

#784 = ((28 + 2 - 3 - 1)/2 + 1)^2 *  4
#160 = ((14 + 2 - 3 - 1)/4 + 1)^2 *  10
#80 = ((4 + 2 - 3 - 1)/2 + 1)^2 *   20
# FC
"""
def width_calculator(dim_in, stride, chan_out):
    return ((dim_in + 2 - 4)/stride + 1)**2*chan_out
"""
def image_dim_calculator(width, stride, chan_out):
    ret= (np.sqrt(width / (chan_out)) - 1)*stride+4-2
    assert np.abs(ret - int(ret)) < 0.4
    return int(ret)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
matplotlib_config()


############################################### Load the data
composed_transform = transforms.Compose([\
    transforms.ToTensor()])

mnist_set = datasets.MNIST(root = './data', train=False, download=True,
    transform=composed_transform)
data_loader = torch.utils.data.DataLoader(mnist_set, 
    batch_size = BATCH_SIZE_TRAIN,
    shuffle=True)


############################################### Fit PED then tSNE
def normalise(inp):
    #inp = np.exp(inp)/(np.exp(inp)+1)
    inp = inp - np.amin(inp)
    inp = inp / np.amax(inp)
    return inp
# First layer
ped = PrincipalEquilibriumDimensions(50)

new_x, labels = ped.fit_transform(data_loader, lamb=LAMBDA, basis=BASIS,
    dist=DIST, weight_decay=2, num_epochs=NUM_EPOCHS, plot_bool=False)

# Get output and normalise
print(new_x)
print(np.amax(new_x))
print(np.amin(new_x))
print(np.mean(new_x))
#plt.hist(new_x.flatten())
#plt.savefig('hist' + str(layer_size) + '.png')
#plt.close()
new_x = normalise(new_x)

#plt.hist(new_x.flatten())
#plt.savefig('hist_normalised' + str(layer_size) + '.png')
#plt.close()
    
new_x = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(new_x)
plt.scatter(new_x[:,0], new_x[:,1], c=labels, s=1)
plt.savefig('ped.png')
plt.close()

#################################################### Fit PCA then tSNE
scaler = StandardScaler()
data_scaled = scaler.fit_transform(mnist_set.data.numpy().reshape((-1,784)))
new_x = PCA(n_components=50).fit_transform(data_scaled)
new_x = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(new_x)
plt.scatter(new_x[:,0], new_x[:,1], c=mnist_set.targets.numpy(), s=1)
plt.savefig('pca.png')
plt.close()

