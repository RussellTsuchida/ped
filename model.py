import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from lib.solvers import anderson
import matplotlib.pyplot as plt

EPS = 1e-03 # Avoid division by zero with continuous Bernoulli
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepEncoderLayerFC(nn.Module):
    def __init__(self, layer_widths, lamb = .1,
        act = 'bernoulli', 
        dropout_mode = 'from_latents'):
        """
        layer_widths (list(int)): [input_dim, width1, ..., widthL-1, output_dim]
        """ 
        super().__init__()
        self.layer_widths = layer_widths
        self.fc_or_conv = 'fc'
        self.dropout_mode = dropout_mode
        #self.dropout_mode = 'off'

        self._init_params()

        self.lamb = lamb
        self.T = lambda x: x

        if act == 'bernoulli':
            self.act = torch.sigmoid
        elif act == 'binomial':
            self.act = lambda x: 10*torch.sigmoid(x)
        elif act == 'cts-bernoulli':
            self.act = lambda x: (torch.exp(x)*(x-1)+1)/(x*(torch.exp(x)-1)+EPS)
        elif act == 'gauss':
            self.act = lambda x: x
            self.T = lambda x: np.sqrt(self.lamb)*x
        elif act == 'relu':
            self.act = nn.functional.relu
            self.T = lambda x: np.sqrt(self.lamb)*x
        elif act == 'poisson':
            self.act = torch.exp
        self.act_str = act

        self.output_dim = np.sum(self.layer_widths[1:])

    def _init_params(self):
        # Initialise weights
        self.M_layers = nn.ModuleList([None])
        self.biases = nn.ParameterList([None])#[None]
        for l in range(len(self.layer_widths)-1):
            in_dim = self.layer_widths[l]
            out_dim = self.layer_widths[l+1]
            M_layer = nn.Linear(in_dim, out_dim, bias = False)

            #M_layer.weight = nn.Parameter(M_layer.weight*0.1) # This is actually W.T in
            # the paper
            nn.init.normal_(M_layer.weight, 0, 1/np.sqrt(in_dim))
            #M_layer.weight = nn.Parameter(M_layer.weight)

            # Initialise biases
            bias = torch.empty(in_dim).to(device)
            nn.init.uniform_(bias, -1/np.sqrt(out_dim),
                1./np.sqrt(out_dim))

            self.M_layers.append(M_layer)
            self.biases.append(nn.Parameter(bias))

        self.M_layers.append(None)
        self.biases.append(None)

    def forward(self, f, Y, mode='encoder'):
        """
        Map input Y and output f to output f. f is a collection of zs in every layer
        """
        # Make a list of z's
        z_list = [Y]
        layer_idx0 = 0
        for idx, l in enumerate(self.layer_widths):
            if idx == 0:
                continue

            z = f[:, layer_idx0:l+layer_idx0]
            z_list.append(z)
            layer_idx0 = l
        z_list.append(None)

        z_list_new = []
        for lm1 in range(len(self.layer_widths)-1):
            zlm1 = z_list[lm1]
            zl = z_list[lm1+1]
            zlp1 = z_list[lm1+2]

            Ml = self.M_layers[lm1+1]
            Mlp1 = self.M_layers[lm1+2]
            bl = self.biases[lm1+1]
            blp1 = self.biases[lm1+2]
            
            etal = nn.functional.linear(zl, Ml.weight.T, bias=bl)
            momentl = self.act(etal)

            if not (Mlp1 is None):
                etalp1 = nn.functional.linear(zlp1, Mlp1.weight.T, bias=blp1)
                momentlp1 = self.act(etalp1)
            else:
                momentlp1 = 0
            dropout = 1
            if self.act_str == 'relu':
                if self.dropout_mode == 'from_inputs':
                    dropout = (zlm1 > 0)*1.
                elif self.dropout_mode == 'from_latents':
                    dropout = (etal > 0)*1.
                elif self.dropout_mode == 'off':
                    dropout = 1.

            zl_new = (Ml(self.T(zlm1)*dropout - momentl) +\
                self.T(momentlp1))/self.lamb
            z_list_new.append(zl_new)

        return torch.hstack(z_list_new)


class ExpfamLoss(nn.Module):
    def __init__(self, exp_str='bernoulli', lamb=1.):
        super().__init__()
        self.exp_str = exp_str
        if exp_str == 'bernoulli':
            self.T = lambda y: y
            self.A = lambda eta: torch.logaddexp(torch.zeros_like(eta), eta)
            self.log_h = lambda y : y*0
        elif exp_str == 'binomial':
            self.T = lambda y: y
            self.A = lambda eta: 10* torch.logaddexp(torch.zeros_like(eta), eta)
            self.log_h = lambda y : y*0
        elif exp_str == 'cts-bernoulli':
            self.T = lambda y: y
            self.A = lambda eta: torch.log((torch.exp(eta) - 1)/(eta + EPS))
            self.log_h = lambda y : y*0
        elif exp_str == 'gauss':
            self.T = lambda y: y
            self.A = lambda eta: eta**2/2
            self.log_h = lambda y : -y**2*lamb/2
        elif exp_str == 'relu':
            self.T = lambda y: y*np.sqrt(lamb)
            self.A = lambda eta: nn.functional.relu(eta)**2/2
            self.log_h = lambda y : -y**2*lamb/2
        elif exp_str == 'poisson':
            self.T = lambda y: y 
            self.A = lambda eta: torch.exp(eta)
            self.log_h = lambda y : y*0
        self.lamb = lamb

    def forward(self, target, eta, z_hidden=None, epoch=None):
        factor = 1
        if epoch is None:
            epoch = np.inf
        if z_hidden is None:
            return torch.sum(-self.T(target)*eta + self.A(eta))/torch.numel(target) - \
            torch.sum(self.log_h(target))/torch.numel(target)
        else:
            total_err = 0
            total_num = 0
            for l in range(len(z_hidden)):
                if l == 0:
                    zlm1 = target
                else:
                    zlm1 = z_hidden[l-1] 
                zl = z_hidden[l]
                etal = eta[l]
                new_layer_loss = self.forward(zlm1,
                    etal.reshape(zlm1.shape))*\
                    torch.numel(zlm1) * factor**l

                total_err = total_err + new_layer_loss
            
            # The last layer also needs to add the log prior    
            #(for other layers this is handled through log_h) 
            total_err = total_err + \
                self.lamb*torch.linalg.norm(zl, ord='fro')**2*factor**l
            total_num = total_num + torch.numel(zlm1)
            return total_err/1000

class DeepDEQAutoencoder(nn.Module):
    def __init__(self, encoder, backbone = False):
        super().__init__()
        self.f = encoder
        self.solver = anderson
        self.backbone = backbone
    
    def forward(self, Y):
        thres = 50
        f0 = torch.zeros((Y.shape[0], int(np.prod(self.f.output_dim))), device=Y.device)
        #f0 = torch.tensor(\
        #    np.random.normal(size=(Y.shape[0], int(np.prod(self.f.output_dim)))), 
        #    device=Y.device, dtype=Y.dtype)

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
                    try:
                        torch.cuda.synchronize()   # To avoid infinite recursion
                    except:
                        pass
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = self.solver(lambda y: autograd.grad(new_f_star, f_star, y,
                    retain_graph=True)[0] + grad, \
                    torch.zeros_like(grad), threshold=thres)['result']
                return new_grad

            self.hook = new_f_star.register_hook(backward_hook)

        if self.backbone: # In this case, only worry about returning the last layer
            _, R = torch.qr(self.f.M_layers[-2].weight.T)
            ret = (R @ new_f_star[:,-self.f.layer_widths[-1]:].T).T
            return ret
        return self.calc_all_etas(new_f_star)

    def calc_all_etas(self, z):
        start_idx = 0
        etas = []
        zs = []
        for l_idx, l_width in enumerate(self.f.layer_widths):
            if l_idx == 0:
                continue
            zl = z[:,start_idx:self.f.layer_widths[l_idx]+start_idx]
            etal =\
                nn.functional.linear(zl, 
                self.f.M_layers[l_idx].weight.T, bias=self.f.biases[l_idx])

            etas.append(etal)
            zs.append(zl)
            start_idx = start_idx + self.f.layer_widths[l_idx]

        return etas, zs


class DeepPED(object):
    def __init__(self, layer_widths):
        self.layer_widths = layer_widths

    def _freeze_unfreeze(self, epoch):
        L = len(self.encoder.M_layers)-2
        for l in range(1, L+1):
            bool_ = (epoch >= 10*(l-1))
            for param in self.encoder.M_layers[l].parameters():
                param.requires_grad = bool_
            self.encoder.biases[l].requires_grad = bool_
        
    def fit(self, data_loader, lamb=1,
        dist='bernoulli', weight_decay=0.001, num_epochs=20,
        plot_bool=False, plot_freq=5, lr=0.01):
        features, label = next(iter(data_loader))
        if len(list(features.shape)) == 4:
            dim_input = np.prod(list(features.shape[1:]))
        else:
            dim_input = features.shape[-1]

        # Initialise the model
        self.encoder = DeepEncoderLayerFC(self.layer_widths, 
            act=dist, lamb=lamb)
        self.encoder.to(device)

        self.model = DeepDEQAutoencoder(self.encoder)
        self.model.to(device)
        #self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=lr,
        #    weight_decay=weight_decay)
        params_ = []
        L = len(self.encoder.M_layers)-2
        for l in range(1, L+1):
            params_ = params_ + [{'params': self.encoder.M_layers[l].parameters(),
                                'weight_decay': (self.layer_widths[l] +\
                                self.layer_widths[l-1])*weight_decay/1000}]
            params_ = params_ + [{'params': self.encoder.biases[l]}]
        
        self.optimiser = torch.optim.Adam(params_, lr=lr)
        #self.optimiser = torch.optim.SGD(self.model.parameters(), lr=lr,
        #    weight_decay=weight_decay)

        self.loss = ExpfamLoss(dist, lamb=lamb)

        for epoch in range(num_epochs):
            print(epoch, flush=True)
            #if epoch > 3:
            #    self.model.f.dropout_mode = 'from_latents'
            #self._freeze_unfreeze(epoch)
            self._train(epoch, data_loader)
            if plot_bool and (epoch % plot_freq) == 0:
                self._test(epoch, data_loader, plot_bool=plot_bool)

    def fit_transform(self, data_loader, lamb=1,
        dist='bernoulli', weight_decay=0.001, num_epochs=20, plot_bool=True,
        plot_freq = 10, lr=0.01, data_loader_test=None, layer_out = -1):
        self.fit(data_loader, lamb, dist, weight_decay, num_epochs,
            plot_bool, plot_freq, lr=lr)
        if data_loader_test is None:
            data_loader_test = data_loader
        return self._test(num_epochs, data_loader_test, plot_bool=plot_bool, 
            layer_out=layer_out)

    def _train(self, epoch, data_loader):
        self.model.train()
        for batch_idx, (data, labels) in enumerate(data_loader):
            self.optimiser.zero_grad()
            layer_in = data.to(device)

            eta, f_star = self.model(layer_in)
            loss_eval = self.loss(layer_in, eta, f_star, epoch=epoch)
            loss_eval.backward()
            self.optimiser.step()
        print(loss_eval.item(), flush=True)
        _, R = torch.qr(self.model.f.M_layers[-2].weight.T)
        print(R)
        #_, R = torch.qr(self.model.f.M_layers[-3].weight.T)
        #print(R)
        for i in range(1, len(self.model.f.M_layers)-1):
            W = self.encoder.M_layers[i].weight.T
            print(torch.linalg.norm(W @ W.T, ord=2)/self.encoder.lamb)

    def _test(self, epoch, data_loader, plot_bool=False, layer_out=-1):
        self.model.eval()
        ret = np.empty((0, self.layer_widths[-1]))

        # For some pruposes (e.g. visualisation) it makes sense to orthoganlise the basis
        _, R = torch.qr(self.model.f.M_layers[layer_out-1].weight.T)

        print(R)
        R = R.detach().cpu().numpy() 

        if plot_bool:
            plt.figure(figsize=(10,10))
       
        for batch_idx, (data, labels) in enumerate(data_loader):
            layer_in = data.to(device)
            with torch.no_grad():
                _, f_star = self.model(layer_in)
            
            f_star = f_star[layer_out].detach().cpu().numpy()
            # Orthogonalise the basis
            f_star = (R @ f_star.T).T

            ret = np.vstack((ret, f_star))

            if plot_bool:
                plt.scatter(f_star[:,0], f_star[:,1], s=1)
        if plot_bool:
            plt.savefig('outputs/' + str(epoch) + '.png', bbox_inches='tight')
            plt.close()
        return ret
