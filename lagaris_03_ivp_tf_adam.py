"""Demonstration of use of TensorFlow to solve the lagaris03 ODE IVP."""


import datetime
from math import cos, exp, sin
import os
import platform
import sys

import numpy as np
import tensorflow as tf

from nnde.math.trainingdata import create_training_grid


def Ya_f(x):
    Ya = exp(-x/5)*sin(x)
    return Ya


def dYa_dx_f(x):
    dYa_dx = exp(-x/5)*(cos(x) - 0.2*sin(x))
    return dYa_dx


def d2Ya_dx2_f(x):
    d2Ya_dx2 = -exp(-x/5)*(0.4*cos(x) + 0.96*sin(x))
    return d2Ya_dx2


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
    output_dir = path_noext
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir


def create_training_data(nt):
    x_train = np.array(create_training_grid(nt))
    return x_train


def build_model(H):
    hidden_layer = tf.keras.layers.Dense(
        units=H, use_bias=True,
        activation=tf.keras.activations.sigmoid,
    )
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        use_bias=False,
    )
    model = tf.keras.Sequential([hidden_layer, output_layer])
    return model


if __name__ == '__main__':
    """Solve lagaris_03_ivp."""

    # Number of hidden nodes.
    H = 10

    # Number of training points.
    nt = 11

    # Random number generator seed.
    random_seed = 0

    # Number of training epochs.
    n_epochs = 1000

    # Learning rate.
    learning_rate = 0.01

    print_system_information()

    # Set up the output directory.
    output_dir = create_output_directory()

    # Create and save the training data, then convert to Variable
    # with shape (nt, 1), i.e. a column vector.
    xt = create_training_data(nt)
    np.savetxt(os.path.join(output_dir, 'training_points.dat'), xt)
    xtv = tf.Variable(xt.reshape((nt, 1)), dtype=tf.float32)
    pass

    # Build the model.
    model = build_model(H)

    # Create history variables.
    losses = []
    phist = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(random_seed)

    # Rename the training Variable for convenience.
    x = xtv

    # Train the model.
    print("Hyperparameters: nt = %s, H = %s, n_epochs = %s, learning_rate = %s"
          % (nt, H, n_epochs, learning_rate))
    t_start = datetime.datetime.now()
    print("Training started at", t_start)
    for epoch in range(n_epochs):
        print("Starting epoch %s." % epoch)

        # Run the forward pass.
        with tf.GradientTape() as tape3:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape1:

                    # Compute the network output.
                    N = model(x)

                    # Compute trial solution.
                    y = x + x**2*N

                # Compute the gradient of trial solution the network output wrt
                # the inputs.
                dy_dx = tape1.gradient(y, x)

            # Compute the 2nd order derivative.
            d2y_dx2 = tape2.gradient(dy_dx, x)

            # Compute the estimate of the differential equation.
            G = d2y_dx2 + 0.2*dy_dx + y + 0.2*tf.math.exp(-x/5)*tf.math.cos(x)

            # Compute the loss function.
            L = tf.reduce_sum(G**2)
        
        # Save the current loss.
        losses.append(L.numpy())

        # Compute the gradient of the loss function wrt the network parameters.
        grad = tape3.gradient(L, model.trainable_variables)

        # Save the parameters used in this pass.
        phist.append(
            np.hstack(
                (model.trainable_variables[0][0].numpy(),    # w (1, H) matrix
                 model.trainable_variables[1].numpy(),       # u (H,) row vector
                 model.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )

        # Update the parameters for this pass.
        for (v, d) in zip(model.trainable_variables, grad):
            v.assign_sub(learning_rate*d)
    
        print("Ending epoch %s." % epoch)

    t_stop = datetime.datetime.now()
    print("Training stopped at", t_stop)
    t_elapsed = t_stop - t_start
    print("Total training time was %s seconds." % t_elapsed.total_seconds())

    # Save the parameter history.
    np.savetxt(os.path.join(output_dir, 'phist.dat'), np.array(phist))

    # Compute the trained solution and derivative at the training points.
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            N = model(xtv)
            Yt = x + x**2*N
        dYt_dx = tape1.gradient(Yt, x)
    d2Yt_dx2 = tape2.gradient(dYt_dx, x)
    np.savetxt(os.path.join(output_dir, 'Yt.dat'), Yt.numpy().reshape((nt)))
    np.savetxt(os.path.join(output_dir, 'dYt_dx.dat'), dYt_dx.numpy().reshape((nt)))
    np.savetxt(os.path.join(output_dir, 'd2Yt_dx2.dat'), d2Yt_dx2.numpy().reshape((nt)))

    # Compute and save the analytical solution and derivatives.
    Ya = np.array([Ya_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'Ya.dat'), Ya)
    dYa_dx = np.array([dYa_dx_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'dYa_dx.dat'), dYa_dx)
    d2Ya_dx2 = np.array([d2Ya_dx2_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'d2Ya_dx2.dat'), d2Ya_dx2)

    # Compute and save the error in the trained solution and derivative.
    Y_err = Yt.numpy().reshape((nt)) - Ya
    np.savetxt(os.path.join(output_dir, 'Y_err.dat'), Y_err)
    dY_dx_err = dYt_dx.numpy().reshape((nt)) - dYa_dx
    np.savetxt(os.path.join(output_dir, 'dY_dx_err.dat'), dY_dx_err)
    d2Y_dx2_err = d2Yt_dx2.numpy().reshape((nt)) - d2Ya_dx2
    np.savetxt(os.path.join(output_dir, 'd2Y_dx2_err.dat'), d2Y_dx2_err)
