# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:02:21 2017

@author: leghtas
"""

import matplotlib.pyplot as plt
from alda.alda import *

plt.close('all')
filename = 'sweep_time_power_freq_IQ.dat'
my_alda = ALDa(filename)
my_data = my_alda.data_raw
my_var_names = my_alda.var_names
print(my_var_names)
res = my_alda.data_xy(my_var_names[2], my_var_names[0])
var_names_xy, my_data_xy, my_data_xy_reshaped, var_xy_reshaped = res
print(var_names_xy)
my_alda.plot_xy(my_var_names[2], my_var_names[0])
