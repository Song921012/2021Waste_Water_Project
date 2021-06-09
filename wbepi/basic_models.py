from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SIR():
    """
    Suspected-Infected-Recovered model
    Only for test.

    """

    def __init__(self, ctrl=None, initS=100, initI=1, initR=0,
                 t0=0, dt=0.1, tend=200,
                 nature_birth=0, nature_death=0,
                 beta=0.2, gamma=0.1):
        # time-dependent function; Initial value; time; parameters;
        self.initvalue = {"initS": initS, "initI": initI, "initR": initR}
        self.timepara = {"t0": t0, "dt": dt, "tend": tend}
        self.para = {"nature_birth": nature_birth, "nature_death": nature_death, "beta": beta, "gamma": gamma}
        self.ctrl = ctrl if ctrl is not None else lambda t: 1

    @staticmethod
    def SIR_model(y, t, nature_birth, nature_death, beta, gamma):
        S, I, R = y
        N = S + I + R
        SIR_1=SIR()
        ctrl = SIR_1.ctrl  # input control function
        return np.array(
            [nature_birth - beta * ctrl(t) * S * I / N - nature_death * S,
             beta * ctrl(t) * S * I / N - gamma * I - nature_death * I,
             gamma * I - nature_death * R])

    # Solving the model by odeint
    def ode_sol(self):
        init_value = [self.initvalue[i] for i in self.initvalue.keys()]
        print("Initial Value:", init_value)
        tspan = np.arange(self.timepara["t0"], self.timepara["tend"], self.timepara["dt"])  # time span
        print("Tspan:", tspan)
        para = tuple([self.para[i] for i in self.para.keys()])  # args
        print("Parameters:", para)
        sol = odeint(self.SIR_model, init_value, tspan, para, )
        return {"tspan": tspan, "solution": sol}

if __name__ == '__main__':
    ctrl = lambda t: 0.1 + 0.1 * np.sin(2 * np.pi * t)
    test_SIR=SIR(ctrl)
    A = test_SIR.ode_sol()
    plt.plot(A["tspan"], A["solution"][:, 2])
    plt.show()