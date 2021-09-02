"""Demonstration of use of TensorFlow to solve the lagaris04 ODE system IVP."""


import datetime
from math import cos, exp, sin
import os
import platform
import sys

import numpy as np
import tensorflow as tf

from nnde.math.trainingdata import create_training_grid


def Ya1_f(x):
    Ya1 = sin(x)
    return Ya1


def dYa1_dx_f(x):
    dYa1_dx = cos(x)
    return dYa1_dx


def Ya2_f(x):
    Ya2 = 1 + x**2
    return Ya2


def dYa2_dx_f(x):
    dYa2_dx = 2*x
    return dYa2_dx


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
    """Solve lagaris_04_ivp."""

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

    # Build the models.
    model1 = build_model(H)
    model2 = build_model(H)

    # Create history variables.
    losses1 = []
    losses2 = []
    phist1 = []
    phist2 = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(random_seed)

    # Rename the training Variable for convenience.
    x = xtv

    # Create the optimizers.
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Train the model.
    print("Hyperparameters: nt = %s, H = %s, n_epochs = %s, learning_rate = %s"
          % (nt, H, n_epochs, learning_rate))
    t_start = datetime.datetime.now()
    print("Training started at", t_start)
    for epoch in range(n_epochs):
        print("Starting epoch %s." % epoch)

        # Run the forward pass.
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:

                # Compute the network outputs.
                N1 = model1(x)
                N2 = model2(x)

                # Compute trial solutions.
                y1 = x*N1
                y2 = 1 + x*N2

            # Compute the gradients of trial solutions.
            dy1_dx = tape1.gradient(y1, x)
            dy2_dx = tape1.gradient(y2, x)

            # Compute the estimates of the differential equations.
            G1 = dy1_dx - tf.math.cos(x) - y1**2 - y2 + 1 + x**2 + tf.math.sin(x)**2
            G2 = dy2_dx - 2*x + (1 + x**2)*tf.math.sin(x) + y1*y2

            # Compute the loss functions.
            L1 = tf.reduce_sum(G1**2)
            L2 = tf.reduce_sum(G2**2)
            L = L1 + L2

        # Save the current losses.
        losses1.append(L1.numpy())
        losses2.append(L2.numpy())

        # Compute the gradients of the loss function wrt the network parameters.
        grad1 = tape2.gradient(L, model1.trainable_variables)
        grad2 = tape2.gradient(L, model2.trainable_variables)

        # Save the parameters used in this pass.
        phist1.append(
            np.hstack(
                (model1.trainable_variables[0][0].numpy(),       # w (1, H) matrix
                    model1.trainable_variables[1].numpy(),       # u (H,) row vector
                    model1.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )
        phist2.append(
            np.hstack(
                (model2.trainable_variables[0][0].numpy(),       # w (1, H) matrix
                    model2.trainable_variables[1].numpy(),       # u (H,) row vector
                    model2.trainable_variables[2][:, 0].numpy()) # v (H, 1) column vector
            )
        )

        # Update the parameters for this pass.
        optimizer1.apply_gradients(zip(grad1, model1.trainable_variables))
        optimizer2.apply_gradients(zip(grad2, model2.trainable_variables))

        print("Ending epoch %s." % epoch)

    t_stop = datetime.datetime.now()
    print("Training stopped at", t_stop)
    t_elapsed = t_stop - t_start
    print("Total training time was %s seconds." % t_elapsed.total_seconds())

    # Save the parameter history.
    np.savetxt(os.path.join(output_dir, 'phist1.dat'), np.array(phist1))
    np.savetxt(os.path.join(output_dir, 'phist2.dat'), np.array(phist2))

    # Compute the trained solution and derivative at the training points.
    with tf.GradientTape(persistent=True) as tape1:
        N1 = model1(xtv)
        N2 = model2(xtv)
        Yt1 = x*N1
        Yt2 = 1 + x*N2
    dYt1_dx = tape1.gradient(Yt1, x)
    dYt2_dx = tape1.gradient(Yt2, x)
    np.savetxt(os.path.join(output_dir, 'Yt1.dat'), Yt1.numpy().reshape((nt)))
    np.savetxt(os.path.join(output_dir, 'dYt1_dx.dat'), dYt1_dx.numpy().reshape((nt)))
    np.savetxt(os.path.join(output_dir, 'Yt2.dat'), Yt2.numpy().reshape((nt)))
    np.savetxt(os.path.join(output_dir, 'dYt2_dx.dat'), dYt2_dx.numpy().reshape((nt)))

    # Compute and save the analytical solution and derivatives.
    Ya1 = np.array([Ya1_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'Ya1.dat'), Ya1)
    Ya2 = np.array([Ya2_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'Ya2.dat'), Ya2)
    dYa1_dx = np.array([dYa1_dx_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'dYa1_dx.dat'), dYa1_dx)
    dYa2_dx = np.array([dYa2_dx_f(x) for x in xt])
    np.savetxt(os.path.join(output_dir,'dYa2_dx.dat'), dYa2_dx)

    # Compute and save the error in the trained solution and derivative.
    Y1_err = Yt1.numpy().reshape((nt)) - Ya1
    np.savetxt(os.path.join(output_dir, 'Y1_err.dat'), Y1_err)
    Y2_err = Yt2.numpy().reshape((nt)) - Ya2
    np.savetxt(os.path.join(output_dir, 'Y2_err.dat'), Y2_err)
    dY1_dx_err = dYt1_dx.numpy().reshape((nt)) - dYa1_dx
    np.savetxt(os.path.join(output_dir, 'dY1_dx_err.dat'), dY1_dx_err)
    dY2_dx_err = dYt2_dx.numpy().reshape((nt)) - dYa2_dx
    np.savetxt(os.path.join(output_dir, 'dY2_dx_err.dat'), dY2_dx_err)
