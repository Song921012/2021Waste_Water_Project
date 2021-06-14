from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from wbepi import basic_models as md
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Parameters, minimize, report_fit
import pandas as pd

# Method One: Nonlinear Least Square Method

## (option) test_data generation

para_test_model = md.SIR(beta=0.2, gamma=0.1, t0=0, dt=5, tend=150)
test_data = para_test_model.ode_sol()
plt.figure(1)
sns.set_theme(style="darkgrid")
plt.plot(test_data["tspan"], test_data["solution"][:, 1])
plt.show()

## parameter estimation by using lmfit
para_estimated = Parameters() #https://lmfit.github.io/lmfit-py/parameters.html
para_estimated.add('beta', value=0.01, min=0, max=1)
para_estimated.add('gamma', value=0.02, min=0, max=1)

"""
from lmfit import Parameters, minimize, report_fit

para_test=Parameters()
para_test.add_many(('amp', 10, True, None, None, None, None),
                ('cen', 4, True, 0.0, None, None, None),
                ('wid', 1, False, None, None, None, None),
                ('frac', 0.5))
A1={"te":1}.items()
A2=para_test.valuesdict().items()
print(type(A2)==type(A1))
para={key:value for key, value in A2}
print(para)
"""

# define error function
def error(para):
    para_model = md.SIR(beta=para["beta"], gamma=para["gamma"], t0=0, dt=5, tend=150)
    model_data = para_model.ode_sol()
    mse = model_data["solution"][:, 1] - test_data["solution"][:, 1]  # only data-data needed
    return mse


out = minimize(error, para_estimated)
report_fit(out.params)
print(error(out.params))
# Show fitting results
result_model = md.SIR(beta=out.params["beta"], gamma=out.params["gamma"], t0=0, dt=1, tend=150)
result_data = result_model.ode_sol()

plt.figure(2)
sns.set_theme(style="darkgrid")
plt.plot(test_data["tspan"], test_data["solution"][:, 1], "o")
plt.plot(result_data["tspan"], result_data["solution"][:, 1])
plt.show()
