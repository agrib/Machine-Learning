import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
from datetime import datetime
import math
import matplotlib.pyplot as plt

#TODO change the file names to your gd avd scd graphdata
# here 'graph_d.npy' is of format [obj_val, time_elapsed, theo_time]

gd = np.load('model_GD.npy')
scd = np.load('model_SCD.npy')
gdtime_elapsed = np.load('timeelapsed_GD.npy')
gdtheotime_vals = np.load('theoriticalval_GD.npy')
gdobj_val = np.load('obective_GD.npy') 
stime_elapsed = np.load('timeelapsed_SCD.npy')
stheotime_vals = np.load('theoriticalval_SCD.npy')
sobj_val = np.load('obective_SCD.npy')

plt.plot(gdtime_elapsed, gdobj_val, marker='o')
plt.plot(stime_elapsed, sobj_val, marker='x')

plt.xlabel('Time elapsed --->')
plt.ylabel('f(W) --->')
plt.title('GD vs SCD')
plt.legend(['GD', 'SCD'], loc='upper right', fontsize='small')
plt.show()
plt.close()


plt.plot(stheotime_vals, sobj_val, marker='x')
plt.plot(gdtheotime_vals, gdobj_val, marker='o')

plt.xlabel('Theoretical time --->')
plt.ylabel('f(W) --->')
plt.title('GD vs SCD')
plt.legend(['GD', 'SCD'], loc='upper right', fontsize='small')
plt.show()
plt.close()





