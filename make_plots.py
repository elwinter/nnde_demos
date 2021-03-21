# Standard plot generation script for nnde jobs.


import math as m

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Width and height of a figure (a standard 8.5x11 inch page)
FIGURE_WIDTH_INCHES = 8.5
FIGURE_HEIGHT_INCHES = 11

# <HACK>
H = 10
NT = 11
NX = 11
NY = 11
N_CMAP_COLORS = 16
# </HACK>


# Set to True to use LaTex rendering.
use_latex = True

# def make_ATE_plots(variable_name, variable_label, equation_string, t_labels, a, t, e):
def make_ATE_plots(variable_name, titles, cb_labels, t_labels, a, t, e):
    """Make a triplet of analytical, trained, and error plots."""

    # Compute the number of subplots to make.
    nt = len(t_labels)

    # Compute the number of rows and columns of subplots.
    n_rows = int(m.ceil(m.sqrt(nt)))
    n_cols = int(m.floor(m.sqrt(nt)))

    # Compute the value limits for each data set.
    (a_min, a_max) = (a.min(), a.max())
    (t_min, t_max) = (t.min(), t.max())
    (e_min, e_max) = (e.min(), e.max())

    # Plot the analytical results.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(FIGURE_WIDTH_INCHES,
                                      FIGURE_HEIGHT_INCHES))
    plt.suptitle(titles[0])
    for i in range(nt):
        ax = axes.flat[i]
        im = ax.imshow(a[i].T, origin='lower', extent=[0, 1, 0, 1], vmin=a_min, vmax=a_max,
                       cmap=plt.get_cmap('viridis', N_CMAP_COLORS))
        if i >= (n_rows - 1)*n_cols:
            ax.set_xlabel('x')
        else:
            ax.tick_params(labelbottom=False)
        if i % n_cols == 0:
            ax.set_ylabel('y')
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.05, 0.9, t_labels[i], color='white')
    # Hide unused subplots.
    for i in range(nt, n_rows*n_cols):
        axes.flat[i].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cb_labels[0])
    plt.savefig(variable_name + '_a.pdf')
    plt.savefig(variable_name + '_a.png')
    plt.close()

    # Plot the trained results.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(FIGURE_WIDTH_INCHES,
                                      FIGURE_HEIGHT_INCHES))
    plt.suptitle(titles[1])
    for i in range(nt):
        ax = axes.flat[i]
        im = ax.imshow(t[i].T, origin='lower', extent=[0, 1, 0, 1], vmin=t_min, vmax=t_max,
                       cmap=plt.get_cmap('viridis', N_CMAP_COLORS))
        if i >= (n_rows - 1)*n_cols:
            ax.set_xlabel('x')
        else:
            ax.tick_params(labelbottom=False)
        if i % n_cols == 0:
            ax.set_ylabel('y')
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.05, 0.9, t_labels[i], color='white')
    # Hide unused subplots.
    for i in range(nt, n_rows*n_cols):
        axes.flat[i].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cb_labels[1])
    plt.savefig(variable_name + '_t.pdf')
    plt.savefig(variable_name + '_t.png')
    plt.close()

    # Plot the absolute error in the trained results.
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(FIGURE_WIDTH_INCHES,
                                      FIGURE_HEIGHT_INCHES))
    plt.suptitle(titles[2])
    for i in range(nt):
        ax = axes.flat[i]
        im = ax.imshow(e[i].T, origin='lower', extent=[0, 1, 0, 1], vmin=e_min, vmax=e_max,
                       cmap=plt.get_cmap('viridis', N_CMAP_COLORS))
        if i >= (n_rows - 1)*n_cols:
            ax.set_xlabel('x')
        else:
            ax.tick_params(labelbottom=False)
        if i % n_cols == 0:
            ax.set_ylabel('y')
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.05, 0.9, t_labels[i], color='white')
    # Hide unused subplots.
    for i in range(nt, n_rows*n_cols):
        axes.flat[i].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cb_labels[2])
    plt.savefig(variable_name + '_e.pdf')
    plt.savefig(variable_name + '_e.png')
    plt.close()


def make_phist_plots(variable_name, variable_label, p):
    plt.title('Evolution of %s' % variable_label)
    H = p.shape[1]
    for i in range(H):
        plt.plot(p[:, i], label='%s' % i)
    plt.xlabel('Epoch')
    plt.ylabel(variable_label)
    plt.legend(bbox_to_anchor=(1.07, 0.5), loc='center')
    plt.savefig(variable_name + '_hist.pdf')
    plt.savefig(variable_name + '_hist.png')
    plt.close()


def main():

    # Use LaTex for equation rendering.
    if use_latex:
        plt.rcParams.update({
            'text.usetex': True
        })

    # Load the training data.
    x_train = np.loadtxt('training_points.dat')

    # N.B. RESULT FILES USE COLUMN ORDER (x y t).

    # Load the analytical results.
    Ya = np.loadtxt('Ya.dat').reshape((NT, NX, NY))
    delYa = np.loadtxt('delYa.dat')
    dYa_dx = delYa[:, 0].reshape((NT, NX, NY))
    dYa_dy = delYa[:, 1].reshape((NT, NX, NY))
    dYa_dt = delYa[:, 2].reshape((NT, NX, NY))
    del2Ya = np.loadtxt('del2Ya.dat')
    d2Ya_dx2 = del2Ya[:, 0].reshape((NT, NX, NY))
    d2Ya_dy2 = del2Ya[:, 1].reshape((NT, NX, NY))
    d2Ya_dt2 = del2Ya[:, 2].reshape((NT, NX, NY))

    # Load the trained results.
    Yt = np.loadtxt('Yt.dat').reshape((NT, NX, NY))
    delYt = np.loadtxt('delYt.dat')
    dYt_dx = delYt[:, 0].reshape((NT, NX, NY))
    dYt_dy = delYt[:, 1].reshape((NT, NX, NY))
    dYt_dt = delYt[:, 2].reshape((NT, NX, NY))
    del2Yt = np.loadtxt('del2Yt.dat')
    d2Yt_dx2 = del2Yt[:, 0].reshape((NT, NX, NY))
    d2Yt_dy2 = del2Yt[:, 1].reshape((NT, NX, NY))
    d2Yt_dt2 = del2Yt[:, 2].reshape((NT, NX, NY))

    # Load the errors in the trained results.
    Yt_err = np.loadtxt('Yt_err.dat').reshape((NT, NX, NY))
    delYt_err = np.loadtxt('delYt_err.dat')
    dYt_dx_err = delYt_err[:, 0].reshape((NT, NX, NY))
    dYt_dy_err = delYt_err[:, 1].reshape((NT, NX, NY))
    dYt_dt_err = delYt_err[:, 2].reshape((NT, NX, NY))
    del2Yt_err = np.loadtxt('del2Yt_err.dat')
    d2Yt_dx2_err = del2Yt_err[:, 0].reshape((NT, NX, NY))
    d2Yt_dy2_err = del2Yt_err[:, 1].reshape((NT, NX, NY))
    d2Yt_dt2_err = del2Yt_err[:, 2].reshape((NT, NX, NY))

    # Load the network parameter history.
    phist = np.loadtxt('phist.dat')

    # Create the labels for each time plot.
    t = x_train[::NX*NY, 0]
    t_labels = ['t = %s' % tt for tt in t]

    # Create a MathJax string for the equation.
    equation_string = r'$\frac {\partial \psi} {dt} - 0.1 \left( \frac {\partial^2 \psi} {\partial x^2} + \frac {\partial^2 \psi} {\partial y^2} \right) = 0$'

    # Create plots in a buffer for writing to a file.
    matplotlib.use('Agg')

    # Create plots for the solution of the differential equation
    # and its derivatives.

    titles = [
        'Analytical solution of\n%s' % equation_string,
        'Neural network solution of\n%s' % equation_string,
        'Absolute error in neural network solution of\n%s' % equation_string
    ]
    cb_labels = [r'$\psi_a$', r'$\psi_t$', r'$\psi_t - \psi_a$']
    make_ATE_plots('Y', titles, cb_labels, t_labels, Ya, Yt, Yt_err)

    titles = [
        'Analytical ' + r'$\frac {\partial \psi} {\partial x}$' + ' of\n%s' % equation_string,
        'Neural network ' + r'$\frac {\partial \psi} {\partial x}$' + ' of\n%s' % equation_string,
        'Absolute error in neural network ' + r'$\frac {\partial \psi} {\partial x}$' + ' of\n%s' % equation_string
    ]
    cb_labels = [r'$\frac {\partial \psi_a} {\partial x}$', r'$\frac {\partial \psi_t} {\partial x}$', r'$\frac {\partial \psi_t} {\partial x} - \frac {\partial \psi_a} {\partial x}$']
    make_ATE_plots('dY_dx', titles, cb_labels, t_labels, dYa_dx, dYt_dx, dYt_dx_err)

    titles = [
        'Analytical ' + r'$\frac {\partial \psi} {\partial y}$' + ' of\n%s' % equation_string,
        'Neural network ' + r'$\frac {\partial \psi} {\partial y}$' + ' of\n%s' % equation_string,
        'Absolute error in neural network ' + r'$\frac {\partial \psi_t} {\partial y}$' + ' of\n%s' % equation_string
    ]
    cb_labels = [r'$\frac {\partial \psi_a} {\partial y}$', r'$\frac {\partial \psi_t} {\partial y}$', r'$\frac {\partial \psi_t} {\partial y} - \frac {\partial \psi_a} {\partial y}$']
    make_ATE_plots('dY_dy', titles, cb_labels, t_labels, dYa_dy, dYt_dy, dYt_dy_err)

    titles = [
        'Analytical ' + r'$\frac {\partial \psi} {\partial t}$' + ' of\n%s' % equation_string,
        'Neural network ' + r'$\frac {\partial \psi} {\partial t}$' + ' of\n%s' % equation_string,
        'Absolute error in neural network ' + r'$\frac {\partial \psi_t} {\partial t}$' + ' of\n%s' % equation_string
    ]
    cb_labels = [r'$\frac {\partial \psi_a} {\partial t}$', r'$\frac {\partial \psi_t} {\partial t}$', r'$\frac {\partial \psi_t} {\partial t} - \frac {\partial \psi_a} {\partial t}$']
    make_ATE_plots('dY_dt', titles, cb_labels, t_labels, dYa_dt, dYt_dt, dYt_dt_err)

    titles = [
        'Analytical ' + r'$\frac {\partial^2 \psi} {\partial x^2}$' + ' of\n%s' % equation_string,
        'Neural network ' + r'$\frac {\partial^2 \psi} {\partial x^2}$' + ' of\n%s' % equation_string,
        'Absolute error in neural network ' + r'$\frac {\partial^2 \psi} {\partial x^2}$' + ' of\n%s' % equation_string
    ]
    cb_labels = [r'$\frac {\partial^2 \psi_a} {\partial x^2}$', r'$\frac {\partial^2 \psi_t} {\partial x^2}$', r'$\frac {\partial^2 \psi_t} {\partial x^2} - \frac {\partial^2 \psi_a} {\partial x^2}$']
    make_ATE_plots('d2Y_dx2', titles, cb_labels, t_labels, d2Ya_dx2, d2Yt_dx2, d2Yt_dx2_err)

    titles = [
        'Analytical ' + r'$\frac {\partial^2 \psi} {\partial y^2}$' + ' of\n%s' % equation_string,
        'Neural network ' + r'$\frac {\partial^2 \psi} {\partial y^2}$' + ' of\n%s' % equation_string,
        'Absolute error in neural network ' + r'$\frac {\partial^2 \psi} {\partial y^2}$' + ' of\n%s' % equation_string
    ]
    cb_labels = [r'$\frac {\partial^2 \psi_a} {\partial y^2}$', r'$\frac {\partial^2 \psi_t} {\partial y^2}$', r'$\frac {\partial^2 \psi_t} {\partial y^2} - \frac {\partial^2 \psi_a} {\partial y^2}$']
    make_ATE_plots('d2Y_dy2', titles, cb_labels, t_labels, d2Ya_dy2, d2Yt_dy2, d2Yt_dy2_err)

    titles = [
        'Analytical ' + r'$\frac {\partial^2 \psi} {\partial t^2}$' + ' of\n%s' % equation_string,
        'Neural network ' + r'$\frac {\partial^2 \psi} {\partial t^2}$' + ' of\n%s' % equation_string,
        'Absolute error in neural network ' + r'$\frac {\partial^2 \psi} {\partial t^2}$' + ' of\n%s' % equation_string
    ]
    cb_labels = [r'$\frac {\partial^2 \psi_a} {\partial t^2}$', r'$\frac {\partial^2 \psi_t} {\partial t^2}$', r'$\frac {\partial^2 \psi_t} {\partial t^2} - \frac {\partial^2 \psi_a} {\partial t^2}$']
    make_ATE_plots('d2Y_dt2', titles, cb_labels, t_labels, d2Ya_dt2, d2Yt_dt2, d2Yt_dt2_err)

    # Plot the evolution of the network parameters.
    wxhist = phist[:, 0:H]
    wyhist = phist[:, H:2*H]
    wthist = phist[:, 2*H:3*H]
    uhist = phist[:, 3*H:4*H]
    vhist = phist[:, 4*H:5*H]

    make_phist_plots('wx', r'$w_x$', wxhist)
    make_phist_plots('wy', r'$w_y$', wyhist)
    make_phist_plots('wt', r'$w_t$', wthist)
    make_phist_plots('u', r'$u$', uhist)
    make_phist_plots('v', r'$v$', vhist)


if __name__ == "__main__":
    main()
