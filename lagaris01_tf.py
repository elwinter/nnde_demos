# Import standard Python modules.
import datetime
import importlib
import platform
import sys
import numpy as np
import tensorflow as tf


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


def create_training_data(xmin, xmax, nt):

    # Create the training data as a row vector (nt-length array).
    (xmin, xmax) = xt_range
    xt = np.linspace(xmin, xmax, nt)

    # Convert the training data to a TensorFlow Variable so we can use it to compute
    # derivatives for use in the differential equation.
    xtv = tf.Variable(xt.reshape((nt, 1)), dtype=tf.float32)
    return xtv


def build_model(H, wmin, wmax, umin, umax, vmin, vmax):
    hidden_layer = tf.keras.layers.Dense(
        units=H, use_bias=True,
        activation=tf.keras.activations.sigmoid,
        kernel_initializer=tf.keras.initializers.RandomUniform(wmin, wmax),
        bias_initializer=tf.keras.initializers.RandomUniform(umin, umax)
    )
    output_layer = tf.keras.layers.Dense(
        units=1,
        activation=tf.keras.activations.linear,
        kernel_initializer=tf.keras.initializers.RandomUniform(vmin, vmax),
        use_bias=False
    )
    model = tf.keras.Sequential([hidden_layer, output_layer])
    return model


if __name__ == '__main__':

    # Number of training points and range.
    nt = 11
    xt_range = (0, 1)

    # Number of hidden nodes.
    H = 10

    # Number of training epochs.
    n_epochs = 1000

    # Learning rate.
    learning_rate = 0.01

    # Starting ranges for network parameters.
    w0_range = (-1, 1)
    u0_range = (-1, 1)
    v0_range = (-1, 1)

    # Random number generator seed.
    random_seed = 0

    # Print the system information.
    print_system_information()

    # Import the problem definition.
    eq_name = "lagaris01"
    eq_module = 'nnde.differentialequation.examples.lagaris_01'
    eq = importlib.import_module(eq_module)

    # Create the training data.
    xtv = create_training_data(*xt_range, nt)
    print(xtv)

    # Build the model.
    model = build_model(H, *w0_range, *u0_range, *v0_range)
    print(model)

    # Create history variables.
    losses = []
    phist = []

    # Set the random number seed for reproducibility.
    tf.random.set_seed(random_seed)

    # Rename the training Variable for convenience.
    x = xtv

    # Train the model.
    print("Hyperparameters: nt = %s, H = %s, n_epochs = %s, eta = %s" % (nt, H, n_epochs, learning_rate))
    t_start = datetime.datetime.now()
    print("Training started at", t_start)
    for epoch in range(n_epochs):
        print("Starting epoch %s." % epoch)

        # Run the forward pass.
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:

                # Compute the network output.
                N = model(x)

            # Compute the gradient of the network output wrt the inputs.
            dN_dx = tape1.gradient(N, x)

            # Compute the estimate of the solution to the differential equation.
            y = 1 + x*N

            # Compute the estimate of the differential equation.
            G = x*dN_dx + N + (x + (1 + 3*x**2)/(1 + x + x**3))*y - x**3 - 2*x - x**2*(1 + 3*x**2)/(1 + x + x**3)

            # Compute the loss function.
            L = tf.sqrt(tf.reduce_sum(G**2)/nt)
        losses.append(L.numpy())

        # Compute the gradient of the loss function wrt the network parameters.
        grad = tape2.gradient(L, model.trainable_variables)

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
    
        # print("Ending epoch %s." % epoch)

    t_stop = datetime.datetime.now()
    print("Training stopped at", t_stop)
    t_elapsed = t_stop - t_start
    print("Total training time was %s seconds." % t_elapsed.total_seconds())

    # Compute the trained solution and derivative at the training points.
    with tf.GradientTape() as tape:
        N = model(xtv)
    dN_dx = tape.gradient(N, xtv)
    N = tf.reshape(N, (nt, 1))
    dN_dx = tf.reshape(dN_dx, (nt, 1))
    Ymt = 1 + xtv*N
    dYmt_dx = xtv*dN_dx + N

    # Compute the analytical solution and derivative at the training points.
    dYat_dx = [eq.dYa_dx(x) for x in xtv.numpy().flatten()]
    Yat = [eq.Ya(x) for x in xtv.numpy().flatten()]

    # Compute the error in the trained solution and derivative.
    err = Ymt.numpy().reshape((nt)) - Yat
    derr = dYmt_dx.numpy().reshape((nt)) - dYat_dx

    # Print the final results.
    # print(losses)
    print(err)
    print(derr)
