# Copyright © 2021 Alexander L. Hayes
# MIT License
# Based on: Trench "Elementary Differential Equations", (CC BY-NC-SA 3.0)

"""
Based on numerical solutions from page 98 of the William F. Trench
"Elementary Differential Equations" 2013 Edition, used under the terms of
the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

Table 3.1.1. includes numerical solutions of y′ + 2y = x³e⁻²ˣ, y(0) = 1
evaluated with various step sizes with the Euler method, as well as an exact
solution. The `x` and `Exact` columns are copied below, as well as solutions
using the nnde `delta` and `BFGS` implementations:

| x     | Exact      | `nnde:delta` | `nnde:BFGS` |
| :---: | :--------: | :----------: | :---------: |
| 0.0   | 1.00000000 | 1.00000      | 1.00000     |
| 0.1   | 0.81875122 | 0.82493      | 0.81875     |
| 0.2   | 0.67058817 | 0.67603      | 0.67059     |
| 0.3   | 0.54992298 | 0.55130      | 0.54992     |
| 0.4   | 0.45220466 | 0.44854      | 0.45220     |
| 0.5   | 0.37362755 | 0.36545      | 0.37362     |
| 0.6   | 0.31095290 | 0.29974      | 0.31095     |
| 0.7   | 0.26139894 | 0.24921      | 0.26140     |
| 0.8   | 0.22257072 | 0.21179      | 0.22257     |
| 0.9   | 0.19241203 | 0.18558      | 0.19241     |
| 1.0   | 0.16916910 | 0.16888      | 0.16916     |

---

Original:
    y′ + 2y = x³e⁻²ˣ
    y(0) = 1

Standard Form:
    G(x,y,(dy/dx)) = (dy/dx) + 2y - x³e⁻²ˣ

Derivative w.r.t. y:
    ∂G/∂y = 2

Derivative w.r.t. dy/dx
    ∂G/∂(dy/dx) = 1
"""

import numpy as np
from nnde.differentialequation.ode.ode1ivp import ODE1IVP
from nnde.neuralnetwork.nnode1ivp import NNODE1IVP


def G(x, Y, dY_dx):
    return dY_dx + (2 * Y) - ((x ** 3) * np.exp(-2 * x))

def dG_dY(x, Y, dY_dx):
    return 2

def dG_ddYdx(x, Y, dY_dx):
    return 1

xt = np.linspace(0.0, 1.0, 11)

ode = ODE1IVP()
ode.G = G
ode.ic = 1
ode.dG_dY = dG_dY
ode.dG_ddYdx = dG_ddYdx

net = NNODE1IVP(ode, nhid=10)
net.train(xt, trainalg="BFGS")
Yt = net.run(xt)

print(Yt)
