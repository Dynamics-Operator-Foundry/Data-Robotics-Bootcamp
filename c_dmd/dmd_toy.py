import numpy as np
import matplotlib as plt
import scipy.io



mat = scipy.io.loadmat('./dynamic_mode_decomposition/CYLINDER_ALL.mat')
print(mat['m'][0][0])
print("here")
