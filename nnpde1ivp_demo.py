"""Demo to show how to solve 1st order PDE IVP with nnde."""


import numpy as np

from nnde.differentialequation.pde.pde1ivp import PDE1IVP
from nnde.math.trainingdata import create_training_grid
from nnde.neuralnetwork.nnpde1ivp import NNPDE1IVP


if __name__ == '__main__':

    # Create training data.
    nx = 5
    ny = 5
    xy_train = np.array(create_training_grid([nx, ny]))
    print('The training points are:\n', xy_train)
    N = nx*ny
    assert len(xy_train) == N
    print('A total of %d training points were created.' % N)

    # Options for training
    training_opts = {}
    training_opts['debug'] = True
    training_opts['verbose'] = True
    training_opts['eta'] = 0.01
    training_opts['maxepochs'] = 1000
    H = 5

    # Test each training algorithm on each equation.
    for eq in ('nnde.differentialequation.examples.example_pde1ivp_01',):
        print('Examining %s.' % eq)
        pde1ivp = PDE1IVP(eq)
        print(pde1ivp)

        # Determine the number of dimensions in the problem.
        m = len(pde1ivp.bc)
        print('Differential equation %s has %d dimensions.' % (eq, m))

        # (Optional) analytical solution and derivative
        if pde1ivp.Ya:
            Ya = np.zeros(N)
            for i in range(N):
                Ya[i] = pde1ivp.Ya(xy_train[i])
            print('The analytical solution at the training points is:')
            print(Ya)
        if pde1ivp.delYa:
            delYa = np.zeros((N, m))
            for i in range(N):
                for j in range(m):
                    delYa[i, j] = pde1ivp.delYa[j](xy_train[i])
            print('The analytical gradient at the training points is:')
            print(delYa)

        # Create and train the networks.
        # for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
        #                  'Newton-CG'):
        for trainalg in ('BFGS',):
            print('Training using %s algorithm.' % trainalg)
            net = NNPDE1IVP(pde1ivp, nhid=H)
            print(net)
            np.random.seed(0)  # Use same seed for reproducibility.
            try:
                net.train(xy_train, trainalg=trainalg, opts=training_opts)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, eq))
                print(e)
                continue
            # print(net.res)
            print('The trained network is:')
            print(net)
            Yt = net.run_debug(xy_train)
            print('The trained solution is:')
            print('Yt =', Yt)
            delYt = net.run_gradient_debug(xy_train)
            print('The trained gradient is:')
            print('delYt =', delYt)

            # (Optional) Error in solution and derivative
            if pde1ivp.Ya:
                print('The error in the trained solution is:')
                print(Yt - Ya)
            if pde1ivp.delYa:
                print('The error in the trained gradient is:')
                print(delYt - delYa)
