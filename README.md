# Principal Equilibrium Dimensions
*Code to accompany the paper titled "Principal equilibrium dimensions"*

## Basic Usage

We roughly follow an sklearn API. For example, for shallow PED from input dimensionality 50 to latent dimensionality 2

    ped = DeepPED([50, 2])
    latents = ped.fit_transform(data)
    
Here `data` is a Pytorch `torch.utils.data.DataLoader`. 

## Hyperparameters

All hyperparameters are set in the `fit` or `fit_transform` method, except for the layer widths, which are set at initialisation. For deep PED with 50 dimensional inputs and hierarchical latents of size 30, 15, 2,

     ped = DeepPED([50, 30, 15, 2])
     
Available keyword hyperparameters are
* `lamb` strictly positive float, which is the lambda in the paper. For implementation, all lambda in each layer is equal.
* `dist` a string representing the combination of A and R. Currently implemented choices when R is the identity are 'bernoulli', 'binomial', 'cts-bernoulli', 'gauss' and 'poisson'. Also available is 'relu', which is a Gaussian likelihood and ReLU R.
* `weight_decay` a nonnegative float representing a factor for the amount of weight decay to include. This is equivalent to L2 regularisation i.e. a (truncated) Gaussian prior over the weights. The coefficient of the L2 regulariser in layer l is `weight_decay * layer_width[l-1]`, where `l` is the index of the layer (starting at zero). So for example for a [50, 30, 15, 2] network, the amount of L2 regularisation in the last latent layer parameters is `50*weight_decay`.
* `num_epochs` integer number of training epochs to run Adam optimiser for
* `lr` strictly positive learning rate for Adam optimiser
* `data_loader_test` a Pytorch `torch.utils.data.DataLoader` to project into latent space. If None, use the training data.
* `layer_out` the latent layer to output, with negative integer index. So `-1` is the last layer, `-2` is the second last layer, and so on.
* `plot_freq` and `plot_bool` are for development purposes. During training, plot some debugging every `plot_freq` epochs if `plot_bool` is True.

## Reproducing the results from the paper
We provide a script called `script_synthetic.py`, which includes all the code required to reproduce the results reported in the paper. Each time this script is run, it completes one run of one of the rows in table 4 (of which 100 are reported in the paper). In order to reproduce results, modify the following global variables as desired

    DIST_TRUE = ... # 'relu' or 'poisson' or 'gauss' or 'bernoulli'
    DIMS_TRUE = ... #[50, 2] or [50, 30, 2] or [50, 30, 15, 2]
    
There are some other globals there to play with as well, if you wish. You will need to run each setting 100 times, the execution of which is system dependent (e.g. with a job manager) and left up to the user. An example for a system using SLURM is provided, see `synthetic_run_all.sh` and `synthetic_run.sh`.
