"""Demonstration of use of nnde to solve ODE IVP"""


import numpy as np

from nnde.differentialequation.ode.ode1ivp import ODE1IVP
from nnde.math.trainingdata import create_training_grid
from nnde.neuralnetwork.nnode1ivp import NNODE1IVP


# Options for training
training_opts = {}
training_opts['debug'] = False
training_opts['verbose'] = True
training_opts['maxepochs'] = 1000

# Create training data.
nx = 11
x_train = np.array(create_training_grid(nx))
if training_opts['debug']:
    print('The training points are:\n', x_train)

# Test each training algorithm on each equation.
for eq in (
    'nnde.differentialequation.examples.lagaris_01',
    'nnde.differentialequation.examples.lagaris_02',
):
    if training_opts['verbose']:
        print('Examining %s.' % eq)
    ode1ivp = ODE1IVP(eq)

    # (Optional) analytical solution and derivative
    if ode1ivp.Ya:
        Ya = np.zeros(nx)
        for i in range(nx):
            Ya[i] = ode1ivp.Ya(x_train[i])
        if training_opts['verbose']:
            print('The analytical solution at the training points is:')
            print(Ya)
    if ode1ivp.dYa_dx:
        dYa_dx = np.zeros(nx)
        for i in range(nx):
            dYa_dx[i] = ode1ivp.dYa_dx(x_train[i])
        if training_opts['verbose']:
            print('The analytical derivative at the training points is:')
            print(dYa_dx)

    # Create and train the networks.
    for trainalg in ('delta', 'Nelder-Mead', 'CG', 'BFGS', 'Newton-CG'):
        if training_opts['verbose']:
            print('Training using %s algorithm.' % trainalg)
        net = NNODE1IVP(ode1ivp)
        if training_opts['debug']:
            print(net)
        np.random.seed(0)  # Use same seed for reproducibility.
        try:
            net.train(x_train, trainalg=trainalg, opts=training_opts)
        except (OverflowError, ValueError) as e:
            print('Error using %s algorithm on %s!' % (trainalg, eq))
            print(e)
            continue
        if training_opts['verbose']:
            print('The trained network is:')
            print(net)

        # Run the trained network to generate a solution.
        Yt = net.run(x_train)
        dYt_dx = net.run_derivative(x_train)
        if training_opts['debug']:
            print('The trained solution is:')
            print('Yt =', Yt)
            print('The trained derivative is:')
            print('dYt_dx =', dYt_dx)

        # (Optional) Error in solution and derivative
        if training_opts['verbose']:
            if ode1ivp.Ya:
                print('The error in the trained solution is:')
                print(Yt - Ya)
            if ode1ivp.dYa_dx:
                print('The error in the trained derivative is:')
                print(dYt_dx - dYa_dx)
