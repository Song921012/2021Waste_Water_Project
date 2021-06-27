import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy as sp

# figure, ax subplots()
"""
ax=plt.gca()子图
fig=plt.gcf()画布
"""

# line plot
'''
plt.plot(x, y, # data
linestyle=, lw=,  # linestyle line width
marker=, markeredgecolor, markeredgewidth, alpha,
label=, color=, )

plt.xlabel()
plt.ylabel()

plt.xscale()
plt.yscale()

plt.text()
plt.title()

plt.legend()

plt.xlim() plt.xticks()
plt.ylim() plt.yticks() 轴范围和刻度

plt.spines() 边框

plt.savefig()       
'''

figure,ax=plt.subplots(1,1)
ax.plot(range(5),range(5))
ax.set_xlabel("x")
plt.show()
