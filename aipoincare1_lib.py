"""
This code is written by Nilanjan Banerjee

Based on https://github.com/KindXiaoming/aipoincare
This module implements a library version of the AI Poincare algorithm
published in https://doi.org/10.1103/PhysRevLett.126.180604 with some minor
modifications and refactoring. The module contains the aipoincare class
which takes the trajectory data for processing. Functions are provided to
calculate the explained ratios, effective number of invariants and their
values.

Classes:
- AIPoincare
- PullNN
"""

##############################################################################
# Imports
import numpy as np # General numerics
from sklearn.decomposition import PCA # Principal component analysis
import torch # Neural network training and inference
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import copy # Make independent copies of mutable objects
import time # For timing
import pickle # Saving data

##############################################################################
# Hard coded parameters
RNG_SEED = 0 # Random number generation seed for predictability
PREPROCESS_FLAG = True # Remove linear invariants?
PREPROCESS_EIGENVALUE_THRESHOLD = 0.001 # Threshold for linear invariant
PULLNN_HIDDEN_LAYERS = [128,128] # Pull NN hidden layers
PULLNN_LEAKYRELU_SLOPE = 0 # Slope for the Leaky Rectified Linear Unit
PULLNN_OPTIMIZER_TYPE = 'Adam' # Either Adam or SGD
PULLNN_LEARNING_RATE = 0.01 # Learning rate for optimization 0.001
PULLNN_BATCH_SIZE = 1024 # Batch size for training 1024
PULLNN_TRAIN_ITER = 500 # Number of training iterations
PULLNN_TRAIN_LOGITER = 200 # Log after every given number of iterations
NEFF_NPOINTS = 100 # Number of starting points for estimating Neff
NEFF_WALK_STEPS = 2000 # Number of random walk steps to estimate Neff
NEFF_FORMULA_A = 2

# Set manual seeds
# torch.manual_seed(RNG_SEED) # Set the seed for torch
# np.random.seed(RNG_SEED) # Set the seed for numpy

##############################################################################
# Classes
class PullNN(nn.Module):
    """
    Neural network to learn global shape of manifold based on the Neural
    Empirical Bayes approach. https://jmlr.org/papers/v20/19-216.html
    This class uses the neural network module exposed by pytorch.

    Methods:
    - __init__
    - forward
    """

    def __init__(self, nn_widths):
        """
        __init__(nn_widths)
        Initialize the pull network.

        Parameters:
        - nn_widths: List of number of neurons in each hidden layer.

        Returns: None
        """
        # By default we will use the cpu
        self.device = torch.device('cpu') # Use CPU
        self.cuda = False
        self.mps = False

        # Find acceleration options
        # First figure out whether we have cuda available
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # Use GPU
            self.cuda = True
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # Use Metal on apple
            self.mps = True

        super(PullNN, self).__init__() # Initialize the torch NN module
        self.nlayers = len(nn_widths) # Total number of layers
        # Linear transformation feedforward layers
        self.linears = nn.ModuleList([nn.Linear(nn_widths[i], nn_widths[i+1],\
            device=self.device) for i in range(self.nlayers-1)])

    def forward(self, x):
        """
        forward(x)
        Run the network to generate output.
        
        Parameters:
        - x: List of inputs to the network
        
        Returns:
        - x: List of outputs from the network
        """
        ActivationFunction = nn.LeakyReLU(PULLNN_LEAKYRELU_SLOPE) # Activation function
        for i in range(self.nlayers-2): # Only the hidden layers have activations?
            x = ActivationFunction(self.linears[i](x)) # Transform input
        # Generate output without an activation function
        x = self.linears[self.nlayers-2](x)
        return x


class AIPoincare:
    """
    Conservation law analyzer based on the AI Poincare algorithm.
    See https://doi.org/10.1103/PhysRevLett.126.180604
    
    Public methods:
    - __init__
    - reset
    - save
    - load
    - run
    - get_neff
    - plot_histograms
    - plot_exp_ratio_diagram
    - plot_mseloss
    
    Private methods:
    - __train_pull_network__
    - __infer_explained_ratios__
    - __compute_neff__
    """

    def __init__(self, trajectory_data, hidden_layers = PULLNN_HIDDEN_LAYERS):
        """
        __init__(trajectory_data, hidden_layers = PULLNN_HIDDEN_LAYERS)
        
        Initialize the AI Poincare object and pre-process the dataset.
        
        Parameters:
        - trajectory_data: Numerical data from a specific phase space
                           trajectory. The data should be of the form of a
                           numpy array with shape Npts x Ndim, where Npts is
                           the total number of points in the trajectory and
                           Ndim is the number of phase space dimensions.
        - hidden_layers: List denoting the structure of hidden layer.
                         Eg. [256, 256]

        Returns: None
        """
        # By default we will use the cpu
        self.device = torch.device('cpu') # Use CPU
        self.cuda = False
        self.mps = False
        # Find acceleration options
        # First figure out whether we have cuda available
        if torch.cuda.is_available():
            self.device = torch.device('cuda') # Use GPU
            self.cuda = True
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # Use Metal on apple
            self.mps = True

        # Instantiate the default random number generator
        self.rng = np.random.default_rng()
        
        # Get data attributes
        self.npts = trajectory_data.shape[0] # Number of points in the data
        self.ndim = trajectory_data.shape[1] # Number of dimensions
        
        ##############################
        self.effective_dimensio = [] #
        self.conserved = []          #
        self.eigenvalues = []        #
        ##############################
        
        # Preprocessing
        # Step 1: Scale and center all data
        # Save the mean and std for possible future use
        self.trajectory_data_mean = np.mean(trajectory_data, axis=0)
        self.trajectory_data_std = np.std(trajectory_data, axis=0)
        # Stop if we have NaNs
        assert ~np.isnan(self.trajectory_data_std).any(), "Given data has NaNs."
        # Do the actual scaling and centering
        
        trajectory_data_normalized = (trajectory_data -
                self.trajectory_data_mean[np.newaxis,:])\
                        /self.trajectory_data_std
        
        #trajectory_data_normalized = trajectory_data
        

        # Step 2: Principal Component Analysis to extract linear invariants
        pca = PCA() # Get the PCA object
        self.data = pca.fit_transform(trajectory_data_normalized)

        # Step 3: Remove linear invariants or scale outputs
        if PREPROCESS_FLAG:
            # Calculate the number of eigenvalues above the threshold
            eigs_above_threshold = np.sum(pca.explained_variance_ratio_
                    > PREPROCESS_EIGENVALUE_THRESHOLD)
            # The number of linear invariants
            self.nlininv = 0 #self.ndim - eigs_above_threshold
            # Clip the dataset assuming that PCA arranges the data in order
            # of decreasing eigenvalues.
            self.data = self.data[:,:]  #eigs_above_threshold
            # Scale the data
            self.data_std = np.std(self.data, axis=0)
            self.data = self.data/self.data_std[np.newaxis,:]
        else:
            self.nlininv = 0 # Set the number of linear invariants to 0
            self.data_std = self.trajectory_data_std # Copy over the std
            # Regularize the data using the noise threshold
            self.data = trajectory_data_normalized/(self.data_std[np.newaxis,:]
                    +PREPROCESS_EIGENVALUE_THRESHOLD)
        
        # Set up layer widths of the pull neural network
        self.nn_widths = copy.deepcopy(hidden_layers)
        self.nn_widths.insert(0, self.data.shape[1]) # Input layer width
        self.nn_widths.append(self.data.shape[1]) # Output layer width

        self.reset() # Reset all the training and inference data

    def reset(self):
        """
        reset()
        
        Reset the trained network and stored statistics.
        Parameters: None
        Returns: None
        """
        # Clear the list of length scales L for which we have stored data.
        self.length_scales_list = []
        # Clear all network state data
        self.nn_state_dict_list = []
        # Clear all loss data
        self.nn_mseloss_list = []
        # Clear the explained ratio data
        self.explained_ratios_list = []
        # Clear the effective number of dimensions list
        self.neff_dist_list = []

    def save(self, filename):
        """
        save(filename)

        Save the state of the AIPoincare object as a pickle file.
        Parameters:
        - filename: Name of pickle file.
        
        Returns: None
        """
        # Construct an object to save
        pickle_obj = [self.length_scales_list, self.nn_state_dict_list,
                      self.nn_mseloss_list,self.explained_ratios_list,
                      self.neff_dist_list]
        with open(filename, 'wb') as fpyc: # Open a pickle file for writing
            pickle.dump(pickle_obj, fpyc) # Dump the serialized object

    def load(self, filename):
        """
        load(filename)

        Load the state of the AIPoincare object from a pickle file.
        Parameters:
        - filename: Name of pickle file.
        
        Returns: None
        """
        with open(filename, 'rb') as fpyc: # Open a pickle file for reading
            pickle_obj = pickle.load(fpyc) # Load the serialized object
        # Now unpack the list and save to objects.
        self.length_scales_list = pickle_obj[0]
        self.nn_state_dict_list = pickle_obj[1]
        self.nn_mseloss_list = pickle_obj[2]
        self.explained_ratios_list = pickle_obj[3]
        self.neff_dist_list = pickle_obj[4]
        
    def run(self, Lwalk_arr, verbosity = 0, optimizer_type = PULLNN_OPTIMIZER_TYPE,
            learning_rate = PULLNN_LEARNING_RATE, training_iterations = PULLNN_TRAIN_ITER,
            training_batchsize = PULLNN_BATCH_SIZE, ntrials = NEFF_NPOINTS):
        """
        run(Lwalk_arr, verbosity = 0, optimizer_type = PULLNN_OPTIMIZER_TYPE,
            learning_rate = PULLNN_LEARNING_RATE, training_iterations = PULLNN_TRAIN_ITER,
            training_batchsize = PULLNN_BATCH_SIZE, ntrials = NEFF_NPOINTS)
        
        Wrapper function to run the AIPoincare algorithm. Parameters:
        - Lwalk_arr: Array of normalized length scales for the random walk.
        - verbosity: 0 for no output. 1 for iteration. 2 for timing.
                     3 for timing and training iteration information.
        - optimizer_type: The type of optimizer used to train the network.
                          Possible values are 'Adam' and 'SGD'.
        - learning_rate: Learning rate for changing the network parameters.
        - training_iterations: Number of iterations for training the network.
        - training_batchsize: Number of points to use in a training batch used
                              for one iteration.
        - ntrials: Number of tangent planes to sample in order to estimate
                   the effective number of invariants.

        Returns: None
        """
        # Set the verbosity of the training and inference
        if verbosity > 0:
            verbosity1 = verbosity - 1
        else:
            verbosity1 = 0
        # For each length scale
        for Lwalk in Lwalk_arr:
            if verbosity > 0:
                print(f'Running AI Poincare on L={Lwalk:.2e}')
            # Train the pull network
            self.__train_pull_network__(Lwalk, verbosity1, optimizer_type,
                    learning_rate, training_iterations, training_batchsize)
            # Infer the explained ratios
            self.__infer_explained_ratios__(Lwalk, verbosity1, ntrials)
            # Compute the effective number of dimensions
            self.__compute_neff__(Lwalk)
            print('Done!')

    def get_neff(self):
        """
        get_neff()

        Obtain the best estimate of number of conserved invariants.

        Parameters:
        - None

        Returns:
        - neff_max: Maximum value of neff over all length scales checked.
        - neff_max_std: Statistical uncertainty on neff_max
        """
        # Check that we have neff distributions already computed
        assert len(self.neff_dist_list) > 0, 'First run aipoincare.'
        # Convert the data into a numpy array
        neff_dist_2d = np.array(self.neff_dist_list)
        # Compute the mean of neff for each length scale
        neff_mean_list = np.mean(neff_dist_2d, axis=1)
        # Get the index of the maximum value of neff
        neff_max_index = np.argmax(neff_mean_list)
        # Extract the maximum mean value of neff
        neff_max = neff_mean_list[neff_max_index]
        # Extract the std corresponding to the maximum mean value of neff
        neff_max_std = np.std(self.neff_dist_list[neff_max_index])
        return neff_max, neff_max_std

    def plot_histograms(self, plt, nbins=200, neff_range=(0,2)):
        """
        plot_histograms(plt)

        Plot histogram of explained ratios and effective number of dimensions
        as a function of walk length scales.

        Parameters:
        - plt: Instance of matplotlib.pyplot

        Returns: None
        """
        n_lwalks = len(self.length_scales_list) # Number of walk length scales
        # Find the range of length scales
        log10_lwalk_min = np.log10(np.min(np.array(self.length_scales_list)))
        log10_lwalk_max = np.log10(np.max(np.array(self.length_scales_list)))
        # Space to save histograms
        neff_hist = np.zeros((n_lwalks, nbins))
        exp_ratio_hist = np.zeros((n_lwalks, nbins))
        for ii in range(n_lwalks): # Iterate through length scales
            # Bin the effective number of dimensions
            neff_hist[ii, :], _ = np.histogram(self.neff_dist_list[ii],
                                            bins=nbins, range=neff_range)
            # Bin the explained ratios
            exp_ratio_hist[ii, :], _ = np.histogram(self.explained_ratios_list[ii],
                                            bins=nbins, range=(0.0, 1.0))
        fig, axs = plt.subplots(1,2) # Generate subplots
        fig.set_size_inches(10, 3) # Nice size
        # Now plot the histograms
        axs[0].imshow(exp_ratio_hist.transpose(),
                extent=(log10_lwalk_min, log10_lwalk_max, 0.0, 1.0),
                origin='lower', cmap='gray_r', aspect='auto', norm='symlog')
        axs[1].imshow(neff_hist.transpose(),
                extent=(log10_lwalk_min, log10_lwalk_max, neff_range[0], neff_range[1]),
                origin='lower', cmap='gray_r', aspect='auto', norm='symlog')
        # Add annotations
        axs[0].set_xlabel('$\\log_{10} L$'); axs[0].set_ylabel('$\\epsilon_n$');
        axs[1].set_xlabel('$\\log_{10} L$'); axs[1].set_ylabel('$n_{eff}$');

    def plot_exp_ratio_diagram(self, plt):
        """
        plot_exp_ratio_diagram(plt)

        Plot the explained ratio diagram for the given data set.

        Parameters:
        - plt: Instance of matplotlib.pyplot

        Returns: None
        """
        # First do some data analysis
        n_lwalks = len(self.length_scales_list) # Number of length scales
        # Make space for mean and std for the ratios and neff
        exp_ratios_mean = np.zeros((n_lwalks, self.data.shape[1]))
        exp_ratios_std = np.zeros((n_lwalks, self.data.shape[1]))
        for ii in range(n_lwalks): # Loop over length scales
            # Evaluate mean and std of the explained ratios
            exp_ratios_mean[ii, :] = np.mean(self.explained_ratios_list[ii], axis=0)
            exp_ratios_std[ii, :] = np.std(self.explained_ratios_list[ii], axis=0)
        # Evaluate the mean neff for each Lwalk
        neff_mean = np.mean(np.array(self.neff_dist_list), axis=1)
        
        
        
        
        self.effective_dimensio.append(neff_mean)
        print(neff_mean)
        # Now generate plots
        fig, ax1 = plt.subplots() # Generate subplots
        fig.set_size_inches(5, 3) # Nice size
        for ii in range(exp_ratios_mean.shape[1]): # Loop over each dimension
            ax1.errorbar(self.length_scales_list, exp_ratios_mean[:,ii], exp_ratios_std[:,ii],
                    color='black', marker='o')
        ax1.set_xscale('log'); ax1.set_xlabel('L')
        ax1.set_ylabel('$\\epsilon_n$')
        # Also plot the neff
        ax2 = ax1.twinx()
        ax2.plot(self.length_scales_list, neff_mean, color='red', ls='--')
        ax2.set_ylabel('$n_{eff}$');
        
    def plot_mseloss(self, plt):
        """
        plot_mseloss(plt)

        Plot the training loss function for the given data set.

        Parameters:
        - plt: Instance of matplotlib.pyplot

        Returns: None
        """
        # Generate plots
        fig, ax = plt.subplots() # Generate subplots
        fig.set_size_inches(5, 3) # Nice size
        for nn_mseloss in self.nn_mseloss_list:
            ax.plot(nn_mseloss/nn_mseloss[0], linewidth=0.5, color='black')
        ax.set_yscale('log')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Normalized Loss')

    def __train_pull_network__(self, Lwalk, verbosity=0,
                        optimizer_type = PULLNN_OPTIMIZER_TYPE,
                        learning_rate = PULLNN_LEARNING_RATE,
                        training_iterations = PULLNN_TRAIN_ITER,
                        training_batchsize = PULLNN_BATCH_SIZE):
        """
        __train_pull_network__(Lwalk, verbosity=0,
                        optimizer_type = PULLNN_OPTIMIZER_TYPE,
                        learning_rate = PULLNN_LEARNING_RATE,
                        training_iterations = PULLNN_TRAIN_ITER,
                        training_batchsize = PULLNN_BATCH_SIZE)
        Train pull network with one given length scale.
        
        Parameters:
        - Lwalk: Length scale of the random walk.
        - verbosity: 0 for no output. 1 for timing.
                     2 for timing and iteration information.
        - optimizer_type: The type of optimizer used to train the network.
                          Possible values are 'Adam' and 'SGD'.
        - learning_rate: Learning rate for changing the network parameters.
        - training_iterations: Number of iterations for training the network.
        - training_batchsize: Number of points to use in a training batch used
                              for one iteration.
        
        Returns: None
        """
        # Do some prep
        tic = time.time() # Start time
        self.length_scales_list.append(Lwalk) # Add length scale to the list
        pullnn = PullNN(self.nn_widths) # Set up the network
        criterion = nn.MSELoss() # L2 Norm as our loss function to optimize
        # Set the optimizer
        if optimizer_type == 'Adam':
            optimizer = optim.Adam(pullnn.parameters(),
                    lr = learning_rate)
        else:
            optimizer = optim.SGD(pullnn.parameters(),
                    lr = learning_rate)
        if verbosity > 0:
            print(f'Training pull network L={Lwalk:.2e}')
        nn_mseloss = [] # Set up a list to accumulate L2 norm data

        # Training loop, uses all data?
        for itrain in range(training_iterations):
            # Prepare the network for training step
            pullnn.train()

            # Prepare a training batch
            # Randomly select points in the trajectory
            choices = np.random.choice(self.npts, training_batchsize)
            # Generate random perturbations to the trajectory
            perturb = torch.normal(0, Lwalk, size = (training_batchsize,
                self.data.shape[1]), device=self.device)
            # Prepare the network input with the random walk step
            inputs0 = torch.tensor(self.data[choices], dtype=torch.float,
                                   device=self.device) + perturb

            # Propagate through the network and calculate loss
            outputs = pullnn(inputs0)
            # Shouldn't the network pull back to the original point ??
            # Why is this -perturb which is the distance to the manifold?
            # Turns out this is implemented differently from the paper.
            loss = criterion(outputs, -perturb)

            # Backpropagation
            optimizer.zero_grad() # Reset the gradient
            loss.backward() # Propagate the loss backward
            nn_mseloss.append(loss.item()) # Save the value of the L2 Norm
            optimizer.step() # Actually update the weights

            # Print stuff if we want
            if verbosity > 1 and itrain % PULLNN_TRAIN_LOGITER == 0:
                print(f'Iteration: {itrain} Loss: {loss:.2e}')

        # Save the neural network state for future use
        self.nn_state_dict_list.append(pullnn.state_dict())
        # Save the MSE loss data for future use
        self.nn_mseloss_list.append(np.array(nn_mseloss))

        # We are done!
        if verbosity > 0:
            print(f'Training complete. {time.time() - tic} seconds elapsed.')
       
    def __infer_explained_ratios__(self, Lwalk, verbosity = 0,
                                   ntrials = NEFF_NPOINTS):
        """
        __infer_explained_ratios__(Lwalk, verbosity = 0, ntrials = NEFF_NPOINTS)
        
        Extract explained ratios from saved information.

        Parameters:
        - Lwalk: Length scale of the random walk.
        - verbosity: 0 for nothing. 1 for timing. 2 for timing and iteration
                     information.
        - ntrials: Number of tangent planes to sample in order to estimate
                   the effective number of invariants.
        
        Returns: None
        """
        # Do some checking!
        try:
            # Find the length scale requested
            training_data_index = self.length_scales_list.index(Lwalk)
            # Load the neural network state for the length scale
            nn_state_dict = self.nn_state_dict_list[training_data_index]
        except ValueError:
            raise RuntimeError(f'Valid network state not found for L = {Lwalk}.')
        tic = time.time() # Start time

        if verbosity > 1:
            print(f'Inferring explained ratio L={Lwalk:.2e}')
        pullnn = PullNN(self.nn_widths) # Set up a fresh network
        # Load the state data for inference
        pullnn.load_state_dict(nn_state_dict)
        exp_ratios_single_L = [] # Start making an array
        for jj in range(ntrials): # Iterate to choose random points

            # Randomly select a point in the trajectory
            choice = np.random.choice(self.npts)
            # Generate random perturbations to the point
            perturb = torch.normal(0, Lwalk, size = (NEFF_WALK_STEPS,
                self.data.shape[1]), dtype=torch.float, device=self.device)
            # Create a bunch of points in the neighborhood
            x0 = torch.tensor(self.data[choice], dtype=torch.float,
                                   device=self.device) + perturb
            

            # Try to walk all the points back to the manifold and convert the
            # torch tensor to numpy matrix data where required.
            # The Pull Network calculates the distance to the closest point in the
            # manifold. So to get back to the manifold we need to add x0. This is
            # different from how the paper is phrased. Not sure why this helps.
            pull_back = pullnn(x0).detach() # Detach from the network
            # Pull back all points to the local tangent plane.
            x0 = x0 + pull_back
            if self.cuda or self.mps: # Transfer to cpu if required
                x0 = x0.cpu()
            x0 = x0.numpy()

            # Step 3: Find the dimensionality of the local set of points
            pca = PCA() # Declare a Principal Component Analysis object
            pca.fit(x0) # Try to fit to the trajectory data
            svs = pca.singular_values_ # Get the eigenvalues
            self.eigenvalues.append(svs)
            
            
            # Fractional contribution
            exp_ratios_single_L.append(svs**2/np.sum(svs**2))
        
        # Convert to a numpy array for easy access
        exp_ratios_single_L = np.array(exp_ratios_single_L)
        self.explained_ratios_list.append(exp_ratios_single_L)

        # We are done!
        if verbosity > 0:
            print(f'Inference complete. {time.time() - tic} seconds elapsed.')

    def __compute_neff__(self, Lwalk):
        """
        __compute_neff__(Lwalk)

        Compute the statistics of effective number of dimensions in the system.

        Parameters:
        - Lwalk: Length scale of the random walk.

        Returns: None
        """
        # Do some checking!
        try:
            # Find the length scale requested
            explained_ratios_index = self.length_scales_list.index(Lwalk)
            # Load the explained ratios for the length scale
            explained_ratios = self.explained_ratios_list[explained_ratios_index]
        except ValueError:
            raise RuntimeError(f'Explained ratios not found for L = {Lwalk}.')
        # Extract the dimensionality of phase space
        ndim = explained_ratios.shape[1]
        # Following the formula defined in Eq. (3) in the paper.
        a = NEFF_FORMULA_A
        mask = explained_ratios < 1.0/(a*ndim)
        neff_dist = np.sum(np.cos(0.5*np.pi*ndim*a*explained_ratios)*mask,axis=1)
        # Finally append the data to the list
        self.neff_dist_list.append(neff_dist)