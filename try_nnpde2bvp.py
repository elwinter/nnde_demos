import datetime
import os
import platform
import sys

import numpy as np

from nnde.differentialequation.pde.pde2bvp import PDE2BVP
from nnde.math.trainingdata import create_training_grid2
from nnde.neuralnetwork.nnpde2bvp import NNPDE2BVP


def print_system_information():
    print("System report:")
    print(datetime.datetime.now())
    print("Host name: %s" % platform.node())
    print("OS: %s" % platform.platform())
    print("uname:", platform.uname())
    print("Python version: %s" % sys.version)
    print("Python build:", platform.python_build())
    print("Python compiler: %s" % platform.python_compiler())
    print("Python implementation: %s" % platform.python_implementation())
    print("Python file: %s" % __file__)


def create_output_directory():
    path_noext, ext = os.path.splitext(__file__)
    output_dir = path_noext + ".output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def create_training_data(*n_train):
    x_train = np.array(create_training_grid2(*n_train))
    return x_train

if __name__ == '__main__':

    # Specify the equation to solve.
    eq_module = 'nnde.differentialequation.examples.lagaris_05'
    eq_name = eq_module.split('.')[-1]

    # Specify the training algorithm.
    trainalg = 'delta'

    # Number of hidden nodes.
    H = 10

    # Number of training points.
    nx_train = 11
    ny_train = 11
    nt = nx_train*ny_train

    # Number of training epochs.
    n_epochs = 1000

    # Learning rate.
    learning_rate = 0.01

    # Random number generator seed.
    random_seed = 0

    # Options for training.
    training_opts = {}
    training_opts['debug'] = True
    training_opts['eta'] = learning_rate
    training_opts['maxepochs'] = n_epochs
    training_opts['verbose'] = True

    # print_system_information()

    # Set up the output directory.
    output_dir = create_output_directory()

    # Create and save the training data.
    xt = create_training_data(nx_train, ny_train)
    np.savetxt(os.path.join(output_dir, 'training_points.dat'), xt)

    # Read the equation specification.
    pde2bvp = PDE2BVP(eq_module)

    # Build the neural network.
    net = NNPDE2BVP(pde2bvp, nhid=H)

    # Initialize the pseudorandom number generater.
    np.random.seed(random_seed)

    # Train the network.
    # print("Hyperparameters: nt = %s, H = %s, n_epochs = %s, learning_rate = %s"
    #       % (nt, H, n_epochs, learning_rate))
    # t_start = datetime.datetime.now()
    # print("Training started at", t_start)
    net.train(xt, trainalg=trainalg, opts=training_opts)
    # t_stop = datetime.datetime.now()
    # print("Training stopped at", t_stop)
    # t_elapsed = t_stop - t_start
    # print("Total training time was %s seconds." % t_elapsed.total_seconds())

    # Save the parameter history.
    np.savetxt(os.path.join(output_dir, 'phist.dat'), net.phist)

    # Compute the trained solution and derivative at the training points.
    Yt = net.run(xt)
    np.savetxt(os.path.join(output_dir, 'Yt.dat'), Yt)
    # dYt_dx = net.run_derivative(xt)
    # np.savetxt(os.path.join(output_dir, 'dYt_dx.dat'), dYt_dx)

    # (Optional) Compute and save the analytical solution and derivative.
    if pde2bvp.Ya:
        Ya = np.array([pde2bvp.Ya(x) for x in xt])
        np.savetxt(os.path.join(output_dir,'Ya.dat'), Ya)
#     if ode1ivp.dYa_dx:
#         dYa_dx = np.array([ode1ivp.dYa_dx(x) for x in xt])
#         np.savetxt(os.path.join(output_dir,'dYa_dx.dat'), dYa_dx)

    # (Optional) Compute and save the error in the trained solution and derivative.
    if pde2bvp.Ya:
        Y_err = Yt - Ya
        np.savetxt(os.path.join(output_dir, 'Y_err.dat'), Y_err)
#     if ode1ivp.dYa_dx:
#         dY_dx_err = dYt_dx - dYa_dx
#         np.savetxt(os.path.join(output_dir, 'dY_dx_err.dat'), dY_dx_err)
