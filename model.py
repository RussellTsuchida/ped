import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from lib.solvers import anderson, broyden
from lib.jacobian import jac_loss_estimate

EPS = 1e-03
cts_bernoulli_act = lambda x: (torch.exp(x)*(x-1)+1)/(x*(torch.exp(x)-1)+EPS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderLayerFC(nn.Module):
    def __init__(self, input_dim, output_dim, lamb = .1,
        act = 'bernoulli', basis='ortho'):
        """
        input_dim (int): The size of the input space. Should be larger than 
            output_dim.
        output_dim (int): The size of the low dimensional subspace. Should be 
            smaller than input_dim.
        """ 
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = basis
        self.fc_or_conv = 'fc'

        self._init_params()

        self.lamb = lamb
        if act == 'bernoulli':
            self.act = torch.sigmoid
        elif act == 'cts-bernoulli':
            self.act = cts_bernoulli_act
        elif act == 'gauss':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.functional.relu
        self.act_str = act

        self.T = lambda x: x

    def _init_params(self):
        # Initialise weights
        self.M_layer = nn.Linear(self.input_dim, self.output_dim, bias = False)
        self.M_layer.weight = nn.Parameter(self.M_layer.weight * 0.1)

        # Initialise biases
        self.bias = nn.Parameter(torch.empty(self.input_dim))
        nn.init.uniform_(self.bias, -1/np.sqrt(self.output_dim),
            1/np.sqrt(self.output_dim))

        self._find_B()

    def _find_B(self):
        """
        if 'ortho':
            Find the semi-orthogonal matrix B from M using either an
        elif 'non-ortho':
            Use the same matrix (transposed) for B
        """
        M = self.M_layer.weight 
        if self.mode == 'ortho':
            U, Sigma, Vt = torch.linalg.svd(M, full_matrices=False)
            self.B = Vt.T @ U.T
        elif self.mode == 'non-ortho':
            self.B = M.T

    def forward(self, f, Y):
        """
        Map input Y and output f to output f.
        """
        Y = torch.flatten(Y, start_dim=1)
        
        self._find_B()

        eta = nn.functional.linear(f, self.B, bias=self.bias)
        moment = self.act(eta)
        dropout = 1
        if self.act_str == 'relu':
            dropout = (eta > 0)*1.
        return self.M_layer(self.T(Y)*dropout - moment)/self.lamb


class EncoderLayerConv(nn.Module):
    def __init__(self, input_chan, output_chan, image_shape, 
        stride=2, basis='ortho', lamb = .1, act='bernoulli'):
        """

        """ 
        super().__init__()
        self.image_shape = image_shape
        self.input_chan = input_chan
        self.output_chan = output_chan
        self.stride = stride
        self.fc_or_conv = 'conv'
        self.output_dim = [output_chan, 
            int(np.floor((image_shape[1]+2-4)/stride+1)), 
            int(np.floor((image_shape[2]+2-4)/stride+1))]
        self.mode = basis
        
        self._init_params()

        self.lamb = lamb
        if act == 'bernoulli':
            self.act = torch.sigmoid
        elif act == 'cts-bernoulli':
            self.act = cts_bernoulli_act
        elif act == 'gauss':
            self.act = lambda x: x
        elif act == 'relu':
            self.act = nn.functional.relu
        self.T = lambda x: x
        self.act_str = act

    def _init_params(self):
        self.M_layer= nn.Conv2d(self.input_chan, self.output_chan, kernel_size=4, 
            stride=self.stride, padding=1)
        self.M_layer.weight = nn.Parameter(self.M_layer.weight * 0.1)

        # Initialise biases
        self.bias = nn.Parameter(torch.zeros(self.image_shape))
        #nn.init.uniform_(self.bias, -0.1, 0.1)

        self._find_B()

    def _find_B(self, mode='ortho'):
        """
        Find the semi-orthogonal matrix B from M using either a svd or polar 
        decomposition.
        """
        if mode == 'ortho':
            M_conv = self.M_layer.weight 
            M = self.convmatrix2d(M_conv, self.image_shape)
            U, Sigma, Vt = torch.linalg.svd(M, full_matrices=False)
            self.B = Vt.T @ U.T
            #self.B = M.T/torch.max(Sigma)
            #print(Sigma)
        elif mode == 'non-ortho':
            pass

    def forward(self, f, Y):
        """
        Map input Y and output f to output f.
        """
        if self.mode == 'ortho':
            # TODO: NEED RELU/DROPOUT IMPLEMENTATION HERE        
            self._find_B()
            moment = self.act(nn.functional.linear(f, self.B))
            moment = moment.reshape(Y.shape)
            return self.M_layer(self.T(Y) - moment)/self.lamb
        elif self.mode == 'non-ortho':
            f = f.reshape([-1] + self.output_dim)
            eta = nn.functional.conv_transpose2d(\
                f, self.M_layer.weight,stride=self.stride, 
                padding=1) + self.bias
            eta = eta.reshape(Y.shape)
            moment = self.act(eta)
            dropout = 1
            if self.act_str == 'relu':
                dropout = (eta > 0)*1.#torch.heaviside(eta, torch.zeros_like(eta))

            ret = self.M_layer(self.T(Y)*dropout - moment)/self.lamb
            return ret.reshape((-1, np.prod(self.output_dim)))

    @staticmethod
    def convmatrix2d(kernel, image_shape, padding: int=0, stride: int=2):
        """
        kernel: (out_channels, in_channels, kernel_height, kernel_width, ...)
        image: (in_channels, image_height, image_width, ...)
        padding: assumes the image is padded with ZEROS of the given amount
        in every 2D dimension equally. The actual image is given with unpadded dimension.
        """

        # If we want to pad, request a bigger matrix as the kernel will convolve
        # over a bigger image.
        if padding:
            old_shape = image_shape
            pads = (padding, padding, padding, padding)
            image_shape = (image_shape[0], image_shape[1] + padding*2, image_shape[2]
                           + padding*2)
        else:
            image_shape = tuple(image_shape)
        assert image_shape[0] == kernel.shape[1]
        assert len(image_shape[1:]) == len(kernel.shape[2:])

        #kernel = kernel.to('cpu') # always perform the below work on cpu
        result_dims = torch.div((torch.tensor(image_shape[1:]) -
                       torch.tensor(kernel.shape[2:])), stride, 
                       rounding_mode='floor') + 1
        mat = torch.zeros((kernel.shape[0], *result_dims, *image_shape), 
            device=kernel.device)
        for i in range(mat.shape[1]):
            for j in range(mat.shape[2]):
                mat[:,i,j,:,i*stride:i*stride+kernel.shape[2],j*stride:j*stride+kernel.shape[3]] = kernel
        mat = mat.flatten(0, len(kernel.shape[2:])).flatten(1)

        # Handle zero padding. Effectively, the zeros from padding do not
        # contribute to convolution output as the product at those elements is zero.
        # Hence the columns of the conv mat that are at the indices where the
        # padded flattened image would have zeros can be ignored. The number of
        # rows on the other hand must not be altered (with padding the output must
        # be larger than without). So..

        # We'll handle this the easy way and create a mask that accomplishes the
        # indexing
        if padding:
            mask = torch.nn.functional.pad(torch.ones(old_shape), pads).flatten()
            mask = mask.bool()
            mat = mat[:, mask]

        return mat


class DEQAutoencoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.f = encoder
        self.solver = anderson
    
    def forward(self, Y):
        thres = 50
        f0 = torch.zeros((Y.shape[0], int(np.prod(self.f.output_dim))), device=Y.device)

        # Forward pass
        with torch.no_grad():
            f_star = self.solver(\
                lambda f: self.f(f, Y), f0, threshold=thres)['result']
            #f_star = f_star.reshape([-1] + self.f.output_dim)
            new_f_star = f_star

        # (Prepare for) Backward pass
        if self.training:
            new_f_star = self.f(f_star.requires_grad_(), Y)
            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = self.solver(lambda y: autograd.grad(new_f_star, f_star, y,
                    retain_graph=True)[0] + grad, \
                    torch.zeros_like(grad), threshold=thres)['result']
                return new_grad

            self.hook = new_f_star.register_hook(backward_hook)

        # Now get the canonical parameters, which are later reconstructed 
        if self.f.fc_or_conv == 'fc':
            eta = nn.functional.linear(new_f_star, self.f.B, bias=self.f.bias)
        elif self.f.fc_or_conv == 'conv':
            ret = new_f_star.reshape((\
                new_f_star.shape[0], self.f.output_chan, 
                int(np.sqrt(new_f_star.shape[1]/self.f.output_chan)),
                int(np.sqrt(new_f_star.shape[1]/self.f.output_chan))))
            eta = nn.functional.conv_transpose2d(\
                ret, self.f.M_layer.weight,stride=self.f.stride, padding=1) +\
                self.f.bias
            eta = eta.reshape((ret.shape[0], -1))

        return eta, new_f_star

"""
class DeepDEQAutoencoder(nn.Module):
    def __init__(self, lamb=0.1, act='bernoulli'):
        super().__init__()

        self.encoder1 = EncoderLayerConv(1, 4, [1, 28, 28], lamb=lamb, 
            act=act)
        self.model1 = DEQAutoencoder(self.encoder1)
        self.norm1 = nn.LayerNorm([4, 13, 13])
        # Ouput after reshape (4, 13, 13)

        self.encoder2 = EncoderLayerConv(4, 4, [4, 13, 13], lamb=lamb,
            act = act)
        self.model2 = DEQAutoencoder(self.encoder2)
        self.norm2 = nn.LayerNorm([4, 6, 6])
        # Output after reshape (4, 6, 6)

        self.encoder3 = EncoderLayerFC(144, 2, lamb=lamb, act=act)
        self.model3 = DEQAutoencoder(self.encoder3)
        # Output after reshape (2)

        if act == 'bernoulli':
            self.act = torch.sigmoid
        elif act == 'cts-bernoulli':
            self.act = cts_bernoulli_act
    
    def forward(self, Y):
        # Do the forward pass
        eta1, f1 = self.model1(Y)

        Y2 = self.act(self.norm1(f1.reshape((-1, 4, 13, 13))))
        eta2, f2 = self.model2(Y2)

        Y3 = self.act(self.norm2(f2.reshape((-1, 4, 6, 6)))).reshape((-1, 4*6*6))
        eta3, f3 = self.model3(Y3)

        Y = self.act(eta3)

        eta = nn.functional.linear(Y, self.encoder2.B)
        Y = self.act(eta)

        eta = nn.functional.linear(Y, self.encoder1.B)

        return eta, f3
"""

class ExpfamLoss(nn.Module):
    def __init__(self, exp_str='bernoulli'):
        super().__init__()
        self.exp_str = exp_str
        if exp_str == 'bernoulli':
            self.T = lambda y: y
            self.A = lambda eta: torch.logaddexp(torch.zeros_like(eta), eta)
        elif exp_str == 'cts-bernoulli':
            self.T = lambda y: y
            self.A = lambda eta: torch.log((torch.exp(eta) - 1)/(eta + EPS))
        elif exp_str == 'gauss':
            self.T = lambda y: y
            self.A = lambda eta: eta**2/2
        elif exp_str == 'relu':
            self.T = lambda y: y
            self.A = lambda eta: nn.functional.relu(eta)**2/2

    def __call__(self, target, eta):
        if self.exp_str == 'relu':
            #target = target * (eta >0) * 1. #torch.heaviside(eta, torch.zeros_like(eta))
            eta = (eta > 0) * eta
        return torch.sum(-self.T(target)*eta + self.A(eta))/torch.numel(target)


class PrincipalEquilibriumDimensions(object):
    def __init__(self, n_components, fc_or_conv='fc', conv_in_chan = 1,
        conv_out_chan = 1, conv_im_shape = [1, 28, 28], conv_stride=2):
        self.n_components = n_components
        self.fc_or_conv = fc_or_conv
        self.conv_in_chan = conv_in_chan
        self.conv_out_chan = conv_out_chan
        self.conv_im_shape = conv_im_shape
        self.conv_stride = conv_stride

    def fit(self, data_loader, lamb=1, basis='non-ortho',
        dist='bernoulli', weight_decay=2, num_epochs=20,
        plot_bool=False, plot_freq=5):
        features, label = next(iter(data_loader))
        if len(list(features.shape)) == 4:
            dim_input = np.prod(list(features.shape[1:]))
        else:
            dim_input = features.shape[-1]

        # Initialise the model
        if self.fc_or_conv == 'fc':
            self.encoder = EncoderLayerFC(dim_input, self.n_components, 
                basis=basis, act=dist, lamb=lamb)
        elif self.fc_or_conv == 'conv':
            self.encoder = EncoderLayerConv(self.conv_in_chan, 
                self.conv_out_chan, self.conv_im_shape,  
                stride=self.conv_stride, basis=basis, act=dist, lamb=lamb)
        self.model = DEQAutoencoder(self.encoder)
        self.model.to(device)
        self.optimiser = torch.optim.AdamW(self.model.parameters(), 
            weight_decay=weight_decay)

        self.loss = ExpfamLoss(dist)

        for epoch in range(num_epochs):
            print(epoch, flush=True)
            self._train(epoch, data_loader)
            if plot_bool and (epoch % plot_freq) == 0:
                self._test(epoch, data_loader, plot_bool=plot_bool)

    def fit_transform(self, data_loader, lamb=1, basis='non-ortho',
        dist='bernoulli', weight_decay=2, num_epochs=20, plot_bool=True,
        plot_freq = 10):
        self.fit(data_loader, lamb, basis, dist, weight_decay, num_epochs,
            plot_bool, plot_freq)
        return self._test(num_epochs, data_loader, plot_bool=plot_bool)

    def _train(self, epoch, data_loader):
        self.model.train()
        for batch_idx, (data, labels) in enumerate(data_loader):
            self.optimiser.zero_grad()
            layer_in = data.to(device)

            eta, f_star = self.model(layer_in)
            loss_eval = self.loss(layer_in, eta.reshape(layer_in.shape))
            loss_eval.backward()
            self.optimiser.step()
        print(loss_eval.item(), flush=True)

    def _test(self, epoch, data_loader, plot_bool=False):
        if plot_bool:
            plt.figure(figsize=(10,10))
        self.model.eval()
        ret = np.empty((0, self.n_components))
        ret_labels = np.empty((0,))
        for batch_idx, (data, labels) in enumerate(data_loader):
            layer_in = data.to(device)
            with torch.no_grad():
                _, f_star = self.model(layer_in)

            f_star = f_star.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            if plot_bool:
                if batch_idx == 0:
                    for c in range(10):
                        ix = np.where(labels == c)
                        plt.scatter(f_star[ix,0], f_star[ix,1], s = 1,
                            label=str(c))
                else:
                    for c in range(10):
                        ix = np.where(labels == c)
                        plt.scatter(f_star[ix,0], f_star[ix,1], s = 1)

            ret = np.vstack((ret, f_star))
            ret_labels = np.hstack((ret_labels, labels))
        if plot_bool:
            plt.legend()
            plt.savefig('outputs/' + str(epoch) + '.png', bbox_inches='tight')
            plt.close()
        return ret, ret_labels


if __name__ == "__main__":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from lib.plotters import matplotlib_config
    
    DIM_OUTPUT          = 2
    BATCH_SIZE_TRAIN    = 100
    NUM_EPOCHS          = 300
    LAMBDA              = 0.5
    DIST                = 'relu'
    BASIS               = 'non-ortho'
    LAYERS              = [784, 160, 2]
    STRIDES             = [2, 4, None]
    CHANNELS            = [4, 10, None]
    
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

    def normalise(inp, centre=False):
        #inp = np.exp(inp)/(np.exp(inp)+1)
        if centre:
            inp = inp - np.amin(inp)
        inp = inp / np.amax(inp)
        return inp

    # Load the data
    composed_transform = transforms.Compose([\
        transforms.ToTensor()])

    mnist_set = datasets.MNIST(root = './data', train=False, download=True,
        transform=composed_transform)
    data_loader = torch.utils.data.DataLoader(mnist_set, 
        batch_size = BATCH_SIZE_TRAIN,
        shuffle=True)
    
    # First layer
    plot_bool = False
    for c, layer_size in enumerate(LAYERS):
        stride = STRIDES[c]
        channel_in = CHANNELS[c-1] if c > 0 else 1
        channel_out = CHANNELS[c]

        if layer_size == 2:
            plot_bool = True
            ped = PrincipalEquilibriumDimensions(layer_size)
        else:
            image_dim = image_dim_calculator(layer_size, stride,
                channel_out)
            ped = PrincipalEquilibriumDimensions(layer_size, 'conv',
                channel_in, channel_out, [channel_in,image_dim,image_dim], 
                conv_stride=stride)

        new_x, labels = ped.fit_transform(data_loader, lamb=LAMBDA, basis=BASIS,
            dist=DIST, weight_decay=2, num_epochs=NUM_EPOCHS, plot_bool=plot_bool,
            plot_freq = 1)
        del ped
        torch.cuda.empty_cache()

        # Get output and normalise
        print(new_x)
        print(np.amax(new_x))
        print(np.amin(new_x))
        print(np.mean(new_x))
        plt.hist(new_x.flatten())
        plt.savefig('hist' + str(layer_size) + '.png')
        plt.close()
        new_x = -1*normalise(new_x)

        plt.hist(new_x.flatten())
        plt.savefig('hist_normalised' + str(layer_size) + '.png')
        plt.close()
        
        if c < len(LAYERS) - 2:
            new_image_dim = image_dim_calculator(LAYERS[c+1], STRIDES[c+1],
                CHANNELS[c+1])
            new_x = new_x.reshape((new_x.shape[0], channel_out,
                new_image_dim, new_image_dim))
        else:
            new_x = new_x.reshape((new_x.shape[0], -1))
        new_x = torch.Tensor(new_x)
        labels = torch.Tensor(labels)
        data = torch.utils.data.TensorDataset(new_x, labels)
        data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE_TRAIN)
    """
    # Second layer
    ped = PrincipalEquilibriumDimensions(2)
    new_x, labels = ped.fit_transform(data_loader, lamb=LAMBDA, basis=BASIS,
        dist=DIST, weight_decay=2, num_epochs=NUM_EPOCHS, plot_bool=True, 
        plot_freq=1)
    """
