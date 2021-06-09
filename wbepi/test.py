from wbepi import models as md
import numpy as np
import matplotlib.pyplot as plt

wbe_model=md.SEIARV()
A=wbe_model.ode_sol()
plt.plot(A["tspan"],A["solution"][:,2])
plt.show()