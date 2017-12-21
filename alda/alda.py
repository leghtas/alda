# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:43:55 2017

@author: leghtas
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


class ALDa(object):
    '''
    Load and analyze loops of data
    data has to have the following format
    var_name_1 var_name2 var_name3 I  Q
    val_1_1    val_2_1   val_3_1   i1 q1
    val_1_1    val_2_1   val_3_2   i2 q2
    val_1_1    val_2_2   val_3_1   i3 q3
    val_1_1    val_2_2   val_3_2   i4 q4
    val_1_2    val_2_1   val_3_1   i5 q5
    val_1_2    val_2_1   val_3_2   i6 q6
    val_1_2    val_2_2   val_3_1   i7 q7
    val_1_2    val_2_2   val_3_2   i8 q8

    vals are ordered
    '''

    def __init__(self, filename):
        self.data_raw = np.loadtxt(filename, skiprows=1)
        _firsline = open(filename, "r").readline()
        _firsline = _firsline.replace('\n', '')
        self.var_names = _firsline.split("\t")[:-2]
        self.var_num = len(self.var_names)
        self.var_vals = {}
        self.var_dims = {}
        self.num_variations = np.size(self.data_raw[:, 0])

        for uu, var_name in enumerate(self.var_names):
            _var = sorted(set(self.data_raw[:, uu]))
            self.var_dims[var_name] = len(_var)
            self.var_vals[var_name] = _var

    def data_xy(self, var_x, var_y):
        if var_x not in self.var_names or var_y not in self.var_names:
            raise Exception('%s or %s are not valid variable names ' % (var_x,
                                                                        var_y))
        else:
            nx = self.var_dims[var_x]
            ny = self.var_dims[var_y]

            data_xy = copy.copy(self.data_raw)
            var_names_xy = copy.copy(self.var_names)

            ix = var_names_xy.index(var_x)
            while ix+1 < self.var_num:
                data_xy[:, [ix, ix+1]] = data_xy[:, [ix+1, ix]]
                var_x = var_names_xy[ix]
                var_x_1 = var_names_xy[ix+1]
                var_names_xy[ix] = var_x_1
                var_names_xy[ix+1] = var_x
                ix += 1

            iy = var_names_xy.index(var_y)
            while iy+1 < self.var_num-1:
                data_xy[:, [iy, iy+1]] = data_xy[:, [iy+1, iy]]
                var_y = var_names_xy[iy]
                var_y_1 = var_names_xy[iy+1]
                var_names_xy[iy] = var_y_1
                var_names_xy[iy+1] = var_y
                iy += 1

            dims_xy = []
            for var_name in var_names_xy:
                dims_xy.append(self.var_dims[var_name])

            print(dims_xy)
            for ivar, var_name in enumerate(var_names_xy):
                aa = np.prod(dims_xy[ivar:])
                jj = 0
                print('--- '+ var_name+' --'+str(ivar)+'aa = '+str(aa))
                while (jj+1)*aa <= self.num_variations:
                    #print(jj*aa)
                    #print(jj*aa+aa)
                    args = data_xy[jj*aa:jj*aa+aa, ivar].argsort()
                    data_xy[jj*aa:jj*aa+aa] = data_xy[jj*aa:jj*aa+aa][args, :]
                    jj += 1

            resh1 = copy.copy(dims_xy)
            resh1.append(2)
            resh2 = copy.copy(dims_xy)
            resh2.append(self.var_num)
            data_xy_reshaped = np.reshape(data_xy[:,-2:], tuple(resh1))
            var_xy_reshaped = np.reshape(data_xy[:,:-2], tuple(resh2))
            return var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped

    def plot_xy(self, var_x, var_y):
        res = self.data_xy(var_x, var_y)
        var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped = res
        x = self.var_vals[var_x]
        y = self.var_vals[var_y]
        shape = np.shape(data_xy_reshaped)
        for ii in np.arange(shape[0]):
            I = data_xy_reshaped[ii,:,:,0]
            Q = data_xy_reshaped[ii,:,:,1]
            data_abs = np.abs(I+1j*Q)
            fig, ax = plt.subplots()
            ax.pcolor(x, y, data_abs)
            ax.set_ylabel(var_y)
            ax.set_xlabel(var_x)
            ax.set_title("%s = %s" % (var_names_xy[0], var_xy_reshaped[ii,0,0][0]))
            ax.axis('tight')
