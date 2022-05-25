import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lib.plotters import matplotlib_config
from model import DeepPED
from model import DeepDEQAutoencoder
from model import DeepEncoderLayerFC
import sys
import os
import umap
import time

BATCH_SIZE_TRAIN    = 500
DIST_TRUE           = 'relu'
DIST_MODEL          = DIST_TRUE
NUM_EPOCHS_SUPERVISED   = 100 
DIMS_TRUE           = [50, 2]
DIMS_MODEL          = DIMS_TRUE
if len(DIMS_TRUE) > 2:
    NUM_EPOCHS_UNSUPERVISED = 10
else:
    NUM_EPOCHS_UNSUPERVISED = 30
WEIGHT_DECAY        = 10.*(len(DIMS_MODEL)-1)
if (DIST_MODEL == 'relu') or (DIST_MODEL == 'poisson'): 
    LAMBDA          = 1.
else:
    LAMBDA          = 0.1
BINOMIAL_N          = 10
NUM_POINTS          = 100000 #10000
SCRIPT_RUN_ID       = int(sys.argv[1])
OUTPUT_DIR          = 'outputs/synthetic/' + DIST_TRUE + \
                        str(len(DIMS_MODEL)-1) + '/' + \
                        str(SCRIPT_RUN_ID) + '/'
LR                  = 0.001
SHAPE               = 'shape'
PRETRAIN            = True

try:
    os.makedirs(OUTPUT_DIR)
except:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
matplotlib_config()

def _generate_square(num_points):
    x = np.linspace(-5, 5, int(np.sqrt(num_points)))
    label1 = np.linspace(0, 1, int(np.sqrt(num_points)))
    z = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))]).T
    label2 = np.transpose([np.tile(label1, len(label1)), 
        np.repeat(label1, len(label1))])
    labels = np.hstack((np.zeros((label2.shape[0], 1)), label2))
    labels = torch.Tensor(labels)

    return z, labels

def _generate_shape(num_points):
    z, labels = _generate_square(num_points)
    # Circles
    idx = np.logical_or(np.linalg.norm(z-2, axis=0) < 3, np.linalg.norm(z+3, axis=0)<2)
    # Square
    idx = np.logical_or(idx, np.linalg.norm(z-np.asarray([[4],[-4]]), axis=0, ord=1) < 1)

    z = z[:,idx]
    labels = labels[idx,:]
    
    return z, labels

def _generate_circle(num_points):
    z, labels = _generate_square(num_points)
    idx = (np.linalg.norm(z, axis=0) < 5)
    z = z[:,idx]
    labels = labels[idx,:]

    return z, labels

def generate_data(num_points, batch_size, dims, shape):
    # generate some points 
    if shape == 'square':
        z, labels = _generate_square(num_points)
    elif shape == 'circle':
        z, labels = _generate_circle(num_points)
    elif shape == 'shape':
        z, labels = _generate_shape(num_points)

    dims = dims[::-1]
    z_in = np.copy(z)
    for dim_idx in range(len(dims)-1):
        low_dim = dims[dim_idx]
        big_dim = dims[dim_idx+1]

        # Generate a basis 
        W = np.random.normal(0, 1, (big_dim, low_dim))/(np.sqrt(low_dim))

        eta = (W @ z_in).T # size = (num_points*num_clusters, big_dim)

        if (DIST_TRUE == 'bernoulli'):
            p = 1/(1+np.exp(-eta))
            Y = np.random.binomial(1, p)
        if (DIST_TRUE == 'binomial'):
            p = 1/(1+np.exp(-eta))
            Y = np.random.binomial(BINOMIAL_N, p)
        if (DIST_TRUE == 'poisson'):
            lam = np.exp(eta*0.5)
            Y = np.random.poisson(lam)
        if DIST_TRUE == 'gauss':
            Y = np.random.normal(eta/np.sqrt(LAMBDA), 
                scale=1/np.sqrt(LAMBDA)*np.ones_like(eta))
        if DIST_TRUE == 'cauchy':
            eta = (eta > 0) * eta
            Y = np.random.standard_cauchy(\
                size=eta.shape).astype(np.float32) + eta.astype(np.float32)
        if DIST_TRUE == 'student':
            eta = (eta > 0) * eta
            Y = np.random.standard_t(2, \
                size=eta.shape).astype(np.float32) + eta.astype(np.float32)
        if DIST_TRUE == 'relu':
            eta = (eta > 0) * eta
            Y = np.random.normal(eta/np.sqrt(LAMBDA), 
                scale=1/np.sqrt(LAMBDA)*np.ones_like(eta))

        Y = torch.Tensor(Y)
        z_in = np.copy(Y).T
    print(Y)

    data = torch.utils.data.TensorDataset(Y, labels)
    data_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=False)
    data_loader_test = \
        torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle=False)

    return data_loader, data, Y, labels, z, data_loader_test

def plot_z_space(z1, z2, z3, labels, prefix):
    if not (z3 is None):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z1, z2, z3, c=labels, s=1)
    else:
        plt.scatter(z1, z2, c=labels, s=1)
        plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(OUTPUT_DIR + prefix + '_waxis.png')

    plt.axis('off')
    plt.gcf().axes[0].get_xaxis().set_visible(False)
    plt.gcf().axes[0].get_yaxis().set_visible(False)
    plt.savefig(OUTPUT_DIR + prefix + '_woaxis.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# Load/generate some synthetic data
data_loader, data_set, true_Y, true_labels, true_z, data_loader_test = \
    generate_data(NUM_POINTS, BATCH_SIZE_TRAIN, DIMS_TRUE, SHAPE)

if PRETRAIN:
    ############################################Plot ground truth latents
    z3 = true_z.T[:,2] if DIMS_TRUE[-1] == 3 else None
    plot_z_space(true_z.T[:,0], true_z.T[:,1], z3, true_labels.numpy(), 'gt' + SHAPE)

    #################################################### Fit TSNE then visualise
    """
    print('Applying tSNE...')
    t0 = time.time()
    tsne_z = TSNE(n_components=DIMS_MODEL[-1], learning_rate='auto',
        init='pca').fit_transform(true_Y)
    print('Took ' + str(time.time() - t0) + ' seconds.')
    z3 = tsne_z[:,2] if DIMS_MODEL[-1] == 3 else None
    plot_z_space(tsne_z[:,0], tsne_z[:,1], z3, true_labels.numpy(), 'tsne' + SHAPE)
    """
    #################################################### Fit UMAP then visualise
    print('Applying UMAP...')
    t0 = time.time()
    umap_z = umap.UMAP().fit_transform(true_Y)
    print('Took ' + str(time.time() - t0) + ' seconds.')
    z3 = umap_z[:,2] if DIMS_MODEL[-1] == 3 else None
    plot_z_space(umap_z[:,0], umap_z[:,1], z3, true_labels.numpy(), 'umap' + SHAPE)
    #################################################### Fit PCA then visualise
    print('Applying PCA...')
    t0 = time.time()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(true_Y.numpy())
    pca_z = PCA(n_components=DIMS_MODEL[-1]).fit_transform(data_scaled)
    print('Took ' + str(time.time() - t0) + ' seconds.')
    z3 = pca_z[:,2] if DIMS_MODEL[-1] == 3 else None
    plot_z_space(pca_z[:,0], pca_z[:,1], z3, true_labels.numpy(), 'pca' + SHAPE)

    ############################################### Fit PED then visualise
    print('Applying PED...')
    t0 = time.time()
    ped = DeepPED(DIMS_MODEL)
    ped_z = ped.fit_transform(data_loader, lamb=LAMBDA, 
        dist=DIST_MODEL, weight_decay=WEIGHT_DECAY, num_epochs=NUM_EPOCHS_UNSUPERVISED, 
        plot_bool=False, plot_freq=50, lr=LR, data_loader_test=data_loader_test)
    print('Took ' + str(time.time() - t0) + ' seconds.')
    z3 = ped_z[:,2] if DIMS_MODEL[-1] == 3 else None
    plot_z_space(ped_z[:,0], ped_z[:,1], z3, true_labels.numpy(), 'ped' + SHAPE)

###################################################
# Regression with 3 neural nets. One has a PED backbone,
# The other has a PCA backbone, one with tSNE
def train(epoch, data_loader, model, optimiser, loss):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimiser.zero_grad()
        data = data.to(device); target = target.to(device)

        prediction = model(data)

        loss_eval = loss(prediction, target)
        loss_eval.backward()
        optimiser.step()

def test(epoch, data_loader, model, loss):
    model.eval()
    loss_total = 0
    total_num = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device); target = target.to(device)

        prediction = model(data)
        loss_eval = loss(prediction, target)*data.shape[0]
        loss_total = loss_total + loss_eval
        total_num = total_num + data.shape[0]
    ret = (loss_total/total_num)
    print(ret)
    return ret.cpu().detach().numpy()

def headnet_gen():
    headnet =  nn.Sequential(\
        nn.Linear(DIMS_MODEL[-1], 100), 
        nn.ReLU(), 
        nn.Linear(100, 1))
    return headnet

loss = torch.nn.MSELoss()

# With each representation z of Y, try and find f(z) for the true z
num_train = int(0.8*true_z.shape[1])
fz = torch.Tensor(true_z[0,:] + true_z[1,:])
data = torch.utils.data.TensorDataset(true_Y, fz.reshape((-1,1)))
idx = np.random.permutation(true_z.shape[1])
fz_train = torch.Tensor(fz.reshape((-1,1))[idx[:num_train],:])
fz_test = torch.Tensor(fz.reshape((-1,1))[idx[num_train:],:])
train_set_ = torch.utils.data.Subset(data, idx[:num_train])
test_set_ = torch.utils.data.Subset(data, idx[num_train:])
train_set = torch.utils.data.DataLoader(train_set_, BATCH_SIZE_TRAIN)
test_set = torch.utils.data.DataLoader(test_set_, BATCH_SIZE_TRAIN)

####################### PED Backbone network
encoder = DeepEncoderLayerFC(DIMS_MODEL, lamb = LAMBDA,
    act = DIST_MODEL)

if PRETRAIN:
    backbone = ped.model
    ped.model.backbone = True
else:
    backbone = DeepDEQAutoencoder(encoder, backbone=True)
headnet = headnet_gen()
pedmodel = nn.Sequential(backbone,headnet)
pedmodel.to(device)

optimiser = torch.optim.Adam(pedmodel.parameters())
ped_loss = []
backbone.f.dropout_mode = 'off'
print('Training PED backbone...')
for epoch in range(NUM_EPOCHS_SUPERVISED):
    ped_loss.append(test(epoch, test_set, pedmodel, loss))
    train(epoch, train_set, pedmodel, optimiser, loss)
ped_loss.append(test(NUM_EPOCHS_SUPERVISED, test_set, pedmodel, loss))


##################### PCA backbone network
train_set_ = torch.utils.data.TensorDataset(torch.Tensor(pca_z[idx[:num_train],:]), 
    fz_train.reshape((-1,1)))
test_set_ = torch.utils.data.TensorDataset(torch.Tensor(pca_z[idx[num_train:],:]), 
    fz_test.reshape((-1,1)))
train_set = torch.utils.data.DataLoader(train_set_, BATCH_SIZE_TRAIN)
test_set = torch.utils.data.DataLoader(test_set_, BATCH_SIZE_TRAIN)

#### Now the actual model
headnet = headnet_gen()
headnet.to(device)

optimiser = torch.optim.Adam(headnet.parameters())
pca_loss = []
print('Training PCA backbone...')
for epoch in range(NUM_EPOCHS_SUPERVISED):
    pca_loss.append(test(epoch, test_set, headnet, loss))
    train(epoch, train_set, headnet, optimiser, loss)
pca_loss.append(test(NUM_EPOCHS_SUPERVISED, test_set, headnet, loss))

##################### tSNE backbone network
if NUM_POINTS <= 10000:
    train_set_ = torch.utils.data.TensorDataset(torch.Tensor(tsne_z[idx[:num_train],:]), 
        fz_train.reshape((-1,1)))
    test_set_ = torch.utils.data.TensorDataset(torch.Tensor(tsne_z[idx[num_train:],:]), 
        fz_test.reshape((-1,1)))
    train_set = torch.utils.data.DataLoader(train_set_, BATCH_SIZE_TRAIN)
    test_set = torch.utils.data.DataLoader(test_set_, BATCH_SIZE_TRAIN)

    #### Now the actual model
    headnet = headnet_gen()
    headnet.to(device)

    optimiser = torch.optim.Adam(headnet.parameters())
    tsne_loss = []
    print('Training tSNE backbone...')
    for epoch in range(NUM_EPOCHS_SUPERVISED):
        tsne_loss.append(test(epoch, test_set, headnet, loss))
        train(epoch, train_set, headnet, optimiser, loss)
    tsne_loss.append(test(NUM_EPOCHS_SUPERVISED, test_set, headnet, loss))

##################### UMAP backbone network
train_set_ = torch.utils.data.TensorDataset(torch.Tensor(umap_z[idx[:num_train],:]), 
    fz_train.reshape((-1,1)))
test_set_ = torch.utils.data.TensorDataset(torch.Tensor(umap_z[idx[num_train:],:]), 
    fz_test.reshape((-1,1)))
train_set = torch.utils.data.DataLoader(train_set_, BATCH_SIZE_TRAIN)
test_set = torch.utils.data.DataLoader(test_set_, BATCH_SIZE_TRAIN)

#### Now the actual model
headnet = headnet_gen()
headnet.to(device)

optimiser = torch.optim.Adam(headnet.parameters())
umap_loss = []
print('Training UMAP backbone...')
for epoch in range(NUM_EPOCHS_SUPERVISED):
    umap_loss.append(test(epoch, test_set, headnet, loss))
    train(epoch, train_set, headnet, optimiser, loss)
umap_loss.append(test(NUM_EPOCHS_SUPERVISED, test_set, headnet, loss))

##################### Network that uses the true zs as inputs
z_train = torch.Tensor(true_z[:,idx[:num_train]])
z_test = torch.Tensor(true_z[:,idx[num_train:]])

train_set_ = torch.utils.data.TensorDataset(z_train.T, fz_train.reshape((-1,1)))
test_set_ = torch.utils.data.TensorDataset(z_test.T, fz_test.reshape((-1,1)))
train_set = torch.utils.data.DataLoader(train_set_, BATCH_SIZE_TRAIN)
test_set = torch.utils.data.DataLoader(test_set_, BATCH_SIZE_TRAIN)

#### Now the actual model
headnet = headnet_gen()
headnet.to(device)

optimiser = torch.optim.Adam(headnet.parameters())
original_loss = []
print('Training with true Zs...')
for epoch in range(NUM_EPOCHS_SUPERVISED):
    original_loss.append(test(epoch, test_set, headnet, loss))
    train(epoch, train_set, headnet, optimiser, loss)
original_loss.append(test(NUM_EPOCHS_SUPERVISED, test_set, headnet, loss))


############################### Save data
if NUM_POINTS <= 10000:
    save_data = np.vstack((ped_loss, pca_loss, tsne_loss, umap_loss, original_loss))
else:
    save_data = np.vstack((ped_loss, pca_loss, np.full((len(pca_loss),),np.inf), umap_loss, original_loss))

np.save(OUTPUT_DIR + 'losses.npy', save_data)   
print(save_data)
