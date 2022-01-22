# Copyright © 2021 Alexander L. Hayes
# MIT License
# Based on: Trench "Elementary Differential Equations", (CC BY-NC-SA 3.0)

"""
Based on numerical solutions from page 99 of the William F. Trench
"Elementary Differential Equations" 2013 Edition, used under the terms of
the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

Table 3.1.3. includes numerical solutions of y′ = -2y² + xy + x², y(0) = 1
evaluated with various step sizes with the Euler method, as well as an exact
solution. The `x` and `Exact` columns are copied below, as well as solutions
using the nnde `delta` and `BFGS` implementations:

| x     | Exact      | `nnde:delta` | `nnde:BFGS` |
| :---: | :--------: | :----------: | :---------: |
| 0.0   | 1.00000000 | 1.00000      | 1.00000     |
| 0.1   | 0.83758449 | 0.84373      | 0.83754     |
| 0.2   | 0.72964189 | 0.72591      | 0.72962     |
| 0.3   | 0.65758037 | 0.64379      | 0.65756     |
| 0.4   | 0.61190179 | 0.59371      | 0.61188     |
| 0.5   | 0.58757549 | 0.57147      | 0.58757     |
| 0.6   | 0.58194222 | 0.57263      | 0.58193     |
| 0.7   | 0.59362952 | 0.59284      | 0.59362     |
| 0.8   | 0.62190745 | 0.62809      | 0.62190     |
| 0.9   | 0.66625084 | 0.67478      | 0.66625     |
| 1.0   | 0.72601579 | 0.72985      | 0.72601     |

---

Original:
    y′ = -2y² + xy + x²
    y(0) = 1

Standard Form:
    G(x,y,(dy/dx)) = (dy/dx) + 2y² - xy - x²

Derivative w.r.t. y:
    ∂G/∂y = 4y - x

Derivative w.r.t. dy/dx
    ∂G/∂(dy/dx) = 1
"""


import numpy as np
from nnde.differentialequation.ode.ode1ivp import ODE1IVP
from nnde.neuralnetwork.nnode1ivp import NNODE1IVP


def G(x, Y, dY_dx):
    return dY_dx + (2 * (Y ** 2)) - (x * Y) - (x ** 2)

def dG_dY(x, Y, dY_dx):
    return (4 * Y) - x

def dG_ddYdx(x, Y, dY_dx):
    return 1

xt = np.linspace(0.0, 1.0, 11)

ode = ODE1IVP()
ode.G = G
ode.ic = 1
ode.dG_dY = dG_dY
ode.dG_ddYdx = dG_ddYdx

net = NNODE1IVP(ode, nhid=10)
net.train(xt, trainalg="delta")
Yt = net.run(xt)

print(Yt)
