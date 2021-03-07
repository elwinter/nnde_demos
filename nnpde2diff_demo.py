"""Demo program for solving diffusion problems with nnde"""


from math import sqrt
import sys

import numpy as np

from nnde.differentialequation.pde.pde2diff import PDE2DIFF
from nnde.math.trainingdata import create_training_grid
from nnde.neuralnetwork.nnpde2diff import NNPDE2DIFF


if __name__ == "__main__":

    # Create training data.
    nx = 5
    ny = 5
    nz = 5
    nt = 5
    xt_train = np.array(create_training_grid([nx, nt]))
    xyt_train = np.array(create_training_grid([nx, ny, nt]))
    xyzt_train = np.array(create_training_grid([nx, ny, nz, nt]))

    # Options for training
    training_opts = {}
    training_opts["debug"] = False
    training_opts["verbose"] = True
    training_opts["eta"] = 0.1
    training_opts["maxepochs"] = 1000
    training_opts["use_jacobian"] = False
    H = 5

    # Test each training algorithm on each equation.
    for pde in ("nnde.differentialequation.examples.diff1d_halfsine",):
        print("Examining %s." % pde)

        # Read the equation definition.
        eq = PDE2DIFF(pde)

        # Fetch the dimensionality of the problem.
        m = len(eq.bc)
        print("Differential equation %s has %d dimensions." % (eq, m))

        # Select the appropriate training set.
        if m == 2:
            x_train = xt_train
        elif m == 3:
            x_train = xyt_train
        elif m == 4:
            x_train = xyzt_train
        else:
            print("INVALID PROBLEM DIMENSION: %s" % m)
            sys.exit(1)
        n = len(x_train)

        # Analytical solution (if available)
        Ya = None
        if eq.Ya is not None:
            print("Computing analytical solution at training points.")
            Ya = np.zeros(n)
            for i in range(n):
                Ya[i] = eq.Ya(x_train[i])
            print("Ya =", Ya)

        # Analytical gradient (if available)
        delYa = None
        if eq.delYa is not None:
            print("Computing analytical gradient at training points.")
            delYa = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    delYa[i][j] = eq.delYa[j](x_train[i])
            print("delYa =", delYa)

        # Analytical Laplacian (if available)
        del2Ya = None
        if eq.del2Ya is not None:
            print("Computing analytical Laplacian at training points.")
            del2Ya = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    del2Ya[i][j] = eq.del2Ya[j](x_train[i])
            print("del2Ya =", del2Ya)

        # for trainalg in ("delta", "Nelder-Mead", "Powell", "CG", "BFGS",
        #                  "Newton-CG"):
        for trainalg in ("BFGS",):
            print("Training using %s algorithm." % trainalg)

            # Create and train the neural network.
            net = NNPDE2DIFF(eq, nhid=H)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg, opts=training_opts)
            except (OverflowError, ValueError) as e:
                print("Error using %s algorithm on %s!" % (trainalg, pde))
                print(e)
                print()
                continue

            if net.res:
                print(net.res)
            print("The trained network is:")
            print(net)

            # Run the network to get the trained solution.
            Yt = net.run(x_train)
            print("The trained solution is:")
            print("Yt =", Yt)

            Yt_rmserr = None
            if eq.Ya:
                Yt_err = Yt - Ya
                print("The error in the trained solution is:")
                print("Yt_err =", Yt_err)
                Yt_rmserr = sqrt(np.sum(Yt_err**2)/n)

            delYt = net.run_gradient(x_train)
            print("The trained gradient is:")
            print("delYt =", delYt)

            delYt_rmserr = None
            if eq.delYa:
                delYt_err = delYt - delYa
                print("The error in the trained gradient is:")
                print("delYt_err =", delYt_err)
                delYt_rmserr = sqrt(np.sum(delYt_err**2)/(n*m))

            del2Yt = net.run_laplacian(x_train)
            print("The trained Laplacian is:")
            print("del2Yt =", del2Yt)

            del2Yt_rmserr = None
            if eq.del2Ya:
                del2Yt_err = del2Yt - del2Ya
                print("The error in the trained Laplacian is:")
                print("del2Yt_err =", del2Yt_err)
                del2Yt_rmserr = sqrt(np.sum(del2Yt_err**2)/(n*m))

            if eq.Ya:
                print("RMS error for trained solution is %s." % Yt_rmserr)
            if eq.delYa:
                print("RMS error for trained gradient is %s." % delYt_rmserr)
            if eq.del2Ya:
                print("RMS error for trained Laplacian is %s." % del2Yt_rmserr)

            # Summary report
            if Yt_rmserr and delYt_rmserr and del2Yt_rmserr:
                print(nx, nt, H, Yt_rmserr, delYt_rmserr, del2Yt_rmserr)
