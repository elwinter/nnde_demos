"""Demonstration of use of nnde to solve ODE IVP"""

import os
import numpy as np

from nnde.differentialequation.ode.ode1ivp import ODE1IVP
from nnde.math.trainingdata import create_training_grid
from nnde.neuralnetwork.nnode1ivp import NNODE1IVP

out_dir, _ = os.path.split(__file__)
out_dir = os.path.join(out_dir, 'lagaris01_demo')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Options for training
training_opts = {}
training_opts['debug'] = False
training_opts['verbose'] = True
training_opts['maxepochs'] = 1000

# Set to True to use LaTex rendering.
use_latex = False

# Width and height of a figure (a standard 8.5x11 inch page)
FIGURE_WIDTH_INCHES = 8.5
FIGURE_HEIGHT_INCHES = 11

# Create training data.
nx = 11
x_train = np.array(create_training_grid(nx))
np.savetxt(os.path.join(out_dir,'training_points.dat'), x_train)
if training_opts['debug']:
    print('The training points are:\n', x_train)

# Specify the equation to solve.
eq = 'nnde.differentialequation.examples.lagaris_01'
if training_opts['verbose']:
    print('Solving %s.' % eq)

# Read the equation specification.
ode1ivp = ODE1IVP(eq)

# (Optional) analytical solution and derivative
if ode1ivp.Ya:
    Ya = np.zeros(nx)
    for i in range(nx):
        Ya[i] = ode1ivp.Ya(x_train[i])
    np.savetxt(os.path.join(out_dir,'Ya.dat'), Ya)
    if training_opts['debug']:
        print('The analytical solution at the training points is:')
        print(Ya)
if ode1ivp.dYa_dx:
    dYa_dx = np.zeros(nx)
    for i in range(nx):
        dYa_dx[i] = ode1ivp.dYa_dx(x_train[i])
    np.savetxt(os.path.join(out_dir,'dYa_dx.dat'), dYa_dx)
    if training_opts['debug']:
        print('The analytical derivative at the training points is:')
        print(dYa_dx)

# Specify the training algorithm.
trainalg = 'BFGS'
if training_opts['verbose']:
    print('Training using %s algorithm.' % trainalg)

# Specify the number of hidden nodes.
H = 10

# Create the neural network.
net = NNODE1IVP(ode1ivp, nhid=H)

# Initialize the pseudorandom number generater. Use same seed for
# reproducibility.
np.random.seed(0)

# Train the network.
try:
    net.train(x_train, trainalg=trainalg, opts=training_opts)
except (OverflowError, ValueError) as e:
    print('Error using %s algorithm on %s!' % (trainalg, eq))
    print(e)
if training_opts['verbose']:
    print('Training complete.')
if training_opts['debug']:
    print('The trained network is:')
    print(net)

# Save the parameter history.
np.savetxt(os.path.join(out_dir, 'phist.dat'), net.phist)

# Run the trained network to generate the trained solution.
Yt = net.run(x_train)
np.savetxt(os.path.join(out_dir, 'Yt.dat'), Yt)
dYt_dx = net.run_derivative(x_train)
np.savetxt(os.path.join(out_dir, 'dYt_dx.dat'), dYt_dx)
if training_opts['debug']:
    print('The trained solution is:')
    print('Yt =', Yt)
    print('The trained derivative is:')
    print('dYt_dx =', dYt_dx)

# (Optional) Error in solution and derivative
if ode1ivp.Ya:
    Y_err = Yt - Ya
    np.savetxt(os.path.join(out_dir, 'Y_err.dat'), Y_err)
    if training_opts['debug']:
        print('The error in the trained solution is:')
        print(Y_err)
if ode1ivp.dYa_dx:
    dY_dx_err = dYt_dx - dYa_dx
    np.savetxt(os.path.join(out_dir, 'dY_dx_err.dat'), dY_dx_err)
    if training_opts['debug']:
        print('The error in the trained derivative is:')
        print(dY_dx_err)

# Plot the results.

# Set to True to use LaTex rendering.
use_latex = True

# Width and height of a figure (a standard 8.5x11 inch page)
FIGURE_WIDTH_INCHES = 8.5
FIGURE_HEIGHT_INCHES = 11

use_latex = True

try:
    import matplotlib.pyplot as plt
except:
    print("Matplotlib not found. Not plotting results.")

if use_latex:
    try:
        import matplotlib
        use_latex = matplotlib.checkdep_usetex(use_latex)
    except:
        pass

plt.rcParams.update({'text.usetex': use_latex})

# Load the training data.
x_train = np.loadtxt(os.path.join(out_dir, 'training_points.dat'))

# N.B. RESULT FILES USE COLUMN ORDER (x y t).

# Load the analytical results.
Ya = np.loadtxt(os.path.join(out_dir, 'Ya.dat'))
dYa_dx = np.loadtxt(os.path.join(out_dir, 'dYa_dx.dat'))

# Load the trained results.
Yt = np.loadtxt(os.path.join(out_dir, 'Yt.dat'))
dYt_dx = np.loadtxt(os.path.join(out_dir, 'dYt_dx.dat'))

# Load the errors in the trained results.
Y_err = np.loadtxt(os.path.join(out_dir, 'Y_err.dat'))
dY_dx_err = np.loadtxt(os.path.join(out_dir, 'dY_dx_err.dat'))

# Load the network parameter history.
phist = np.loadtxt(os.path.join(out_dir, 'phist.dat'))

# Create plots in a buffer for writing to a file.
matplotlib.use('Agg')

# Create the figure to hold the solutions and error.
fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
)

# Plot the analytical solution.
plt.subplot(311)
plt.plot(x_train, Ya)
plt.title('Analytical Solution')
plt.ylabel('$\psi_a(x)$')
plt.grid()

# Plot the trained solution.
plt.subplot(312)
plt.plot(x_train, Yt)
plt.title('Trained Solution')
plt.ylabel('$\psi_t(x)$')
plt.grid()

# Plot the error in the trained solution.
plt.subplot(313)
plt.plot(x_train, Y_err)
plt.title('Error in Trained Solution')
plt.ylabel('$\psi_t(x) - \psi_a(x)$')
plt.xlabel('x')
plt.grid()

# Save the figure.
plt.savefig(os.path.join(out_dir, 'Y_ate.png'))

# Create the figure to hold the parameter history plots.
fig, axes = plt.subplots(
    nrows=3, ncols=1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
)

# Exract parameter history.
whist = phist[:, 0:H]
uhist = phist[:, 1*H:2*H]
vhist = phist[:, 2*H:3*H]

# Plot the hidden node weight history.
plt.subplot(311)
for i in range(H):
    plt.plot(whist[:, i], label='$w_%s$' % i)
plt.title('Evolution of hidden node weights')
plt.ylabel('$w_k$')
plt.legend(bbox_to_anchor=(1.07, 0.5), loc='center')

# Plot the hidden node bias history.
plt.subplot(312)
for i in range(H):
    plt.plot(uhist[:, i], label='$u_%s$' % i)
plt.title('Evolution of hidden node biases')
plt.ylabel('$u_k$')
plt.legend(bbox_to_anchor=(1.07, 0.5), loc='center')

# Plot the output node weight history.
plt.subplot(313)
for i in range(H):
    plt.plot(vhist[:, i], label='$v_%s$' % i)
plt.title('Evolution of output node weights')
plt.xlabel('Epoch')
plt.ylabel('$v_k$')
plt.legend(bbox_to_anchor=(1.07, 0.5), loc='center')

# Save the parameter history figure.
plt.savefig(os.path.join(out_dir, 'lagaris01_demo.png'))
