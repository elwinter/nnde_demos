# Import standard Python modules.
import datetime
import importlib
import logging
from math import exp
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# Import the problem definition.
eq_name = "lagaris01"
eq_module = 'nnde.differentialequation.examples.lagaris_01'
eq = importlib.import_module(eq_module)

# Compute the analytical solution and derivative.
# Note that Y is used in place of \psi in the code.
# na = 101
# xa = np.linspace(0, 1, na)
# Ya = np.zeros(na)
# dYa_dx = np.zeros(na)
# for i in range(na):
#     Ya[i] = eq.Ya(xa[i])
#     dYa_dx[i] = eq.dYa_dx(xa[i])

# Plot the analytical solution and derivative.
# plt.plot(xa, Ya, label="$\psi_a$")
# plt.plot(xa, dYa_dx, label="$d\psi_a/dx$")
# plt.xlabel("x")
# plt.ylabel("$\psi_a$ or $d\psi_a/dx$")
# plt.grid()
# plt.legend()
# plt.title("Analytical solution for %s\n(compare to Lagaris et al. (1998), Figure 1(a))" %
#          (eq_name))

# Hyperparameters for the problem and network.

# Number of training points and range.
nt = 11
xt_range = (0, 1)

# Number of hidden nodes.
H = 10

# Training algorithm.
training_algorithm = 'delta'

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
print(datetime.datetime.now())
print("Host name: %s" % platform.node())
print("OS: %s" % platform.platform())
print("uname:", platform.uname())
print("Python version: %s" % sys.version)
print("Python build:", platform.python_build())
print("Python compiler: %s" % platform.python_compiler())
print("Python implementation: %s" % platform.python_implementation())

# Create the training data as a row vector (nt-length array).
(xmin, xmax) = xt_range
xt = np.linspace(xmin, xmax, nt)

# Convert the training data to a TensorFlow Variable so we can use it to compute
# derivatives for use in the differential equation.
xtv = tf.Variable(xt.reshape((nt, 1)), dtype=tf.float32)

# Create the network.
hidden_layer = tf.keras.layers.Dense(
    units=H, use_bias=True,
    activation=tf.keras.activations.sigmoid,
    kernel_initializer=tf.keras.initializers.RandomUniform(*w0_range),
    bias_initializer=tf.keras.initializers.RandomUniform(*u0_range)
)
output_layer = tf.keras.layers.Dense(
    units=1,
    activation=tf.keras.activations.linear,
    kernel_initializer=tf.keras.initializers.RandomUniform(*v0_range),
    use_bias=False
)
model = tf.keras.Sequential([hidden_layer, output_layer])

# Train the network.

# Create history variables.
losses = []
phist = []

# Set the random number seed for reproducibility.
tf.random.set_seed(random_seed)

# Rename the training Variable for convenience.
x = xtv

# Turn off TensorFlow warning messages (for now).
# tf.get_logger().setLevel(logging.ERROR)

# Train the model.
t_start = datetime.datetime.now()
print("Training started at", t_start)

# CHANGE THIS CODE TO USE @tf.function.
for i in range(n_epochs):

    # Compute the forward pass for each training point.
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

t_stop = datetime.datetime.now()
print("Training stopped at", t_stop)
t_elapsed = t_stop - t_start
print("Total training time was %s seconds." % t_elapsed.total_seconds())
print("Hyperparameters: nt = %s, H = %s, n_epochs = %s, eta = %s" % (nt, H, n_epochs, learning_rate))

print("x.shape =", x.shape)
print("w.shape =", model.trainable_variables[0].shape)
print("u.shape =", model.trainable_variables[1].shape)
print("v.shape =", model.trainable_variables[2].shape)
print("N.shape =", N.shape)
print("dN_dx.shape =", dN_dx.shape)
print("y.shape =", y.shape)
print("G.shape =", G.shape)
print("L.shape =", L.shape)

# Compute the trained solution and derivative at the training points.
with tf.GradientTape() as tape:
    N = model(xtv)
dN_dx = tape.gradient(N, xtv)
N = tf.reshape(N, (nt, 1))
dN_dx = tf.reshape(dN_dx, (nt, 1))
Ymt = 1 + xtv*N
dYmt_dx = xtv*dN_dx + N

# Compute the analytical solution and derivative at the training points.
Yat = [eq.Ya(x) for x in xt]
dYat_dx = [eq.dYa_dx(x) for x in xt]

# Compute the error in the trained solution and derivative.
err = Ymt.numpy().reshape((nt)) - Yat
derr = dYmt_dx.numpy().reshape((nt)) - dYat_dx

# Plot the loss function history.
# plt.semilogy(losses)
# plt.xlabel("Epoch")
# plt.ylabel(r"Loss function $\sqrt {\frac {\sum G_i^2} {n_t}}$")
# plt.grid()
# plt.title("Loss function evolution for %s (%s, %s)\n$\eta$=%s, H=%s, $n_t$=%s, $w_0$=%s, $u_0$=%s, $v_0$=%s" %
#           (eq_name, 'TensorFlow', training_algorithm, learning_rate, H, nt, w0_range, u0_range, v0_range))
# plt.show()

# Plot the errors in the trained solution and derivative.
# plt.plot(xt, err, label="$\psi_a$")
# plt.plot(xt, derr, label="$d\psi_a/dx$")
# plt.xlabel("x")
# plt.ylabel("Absolute error in $\psi_a$ or $d\psi_a/dx$")
# plt.grid()
# plt.legend()
# plt.title("Error in trained solution and derivative for %s (%s, %s)\n$\eta$=%s, H=%s, $n_t$=%s, $w_0$=%s, $u_0$=%s, $v_0$=%s" %
#           (eq_name, 'nnde', training_algorithm, learning_rate, H, nt, w0_range, u0_range, v0_range))
# plt.show()

# # Plot the parameter histories.
# phist = np.array(phist)
# plt.figure(figsize=(12, 14))

# # w
# plt.subplot(311)
# plt.plot(phist[:, 0:H])
# plt.title("Hidden weight $w$")
# plt.grid()

# # u
# plt.subplot(312)
# plt.plot(phist[:, H:2*H])
# plt.title("Hidden bias $u$")
# plt.grid()

# # v
# plt.subplot(313)
# plt.plot(phist[:, 2*H:3*H])
# plt.title("Output weight $v$")
# plt.grid()

# plt.suptitle("Parameter evolution for %s (%s, %s)\n$\eta$=%s, H=%s, $n_t$=%s, $w_0$=%s, $u_0$=%s, $v_0$=%s" %
#           (eq_name, 'nnde', training_algorithm, learning_rate, H, nt, w0_range, u0_range, v0_range))
# plt.subplots_adjust(hspace=0.2)
# plt.show()
