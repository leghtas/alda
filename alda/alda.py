# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:43:55 2017

@author: leghtas
"""

import numpy as np
import copy
import matplotlib.pyplot as plt

#import fit

def make_var_vals(base_array, r, L):
    n = int(np.shape(base_array)[0])
    r = int(r)
    L = int(L)
    assert (L/(n*r) == int(L/(n*r)))
    var_vals0 = np.zeros(n*r)
    for ii in np.arange(n):
        var_vals0[r*ii:r*ii+r] = base_array[ii]
    var_vals = var_vals0
    while np.shape(var_vals)[0] != L:
        var_vals = np.append(var_vals, var_vals0)
    return var_vals

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
        self.num_variations_padded = self.num_variations

        for uu, var_name in enumerate(self.var_names):
            _var = sorted(set(self.data_raw[:, uu]))
            self.var_dims[var_name] = len(_var)
            self.var_vals[var_name] = _var

    def data_padded(self):
        i0 = self.num_variations
        i1 = np.prod(list(self.var_dims.values()))
        _m = i1 - i0

        if _m == 0:
            self.num_variations_padded = self.num_variations
            return self.data_raw
        else:
            self.num_variations_padded = i1
            _n = self.var_num + 2

            data_padded = copy.copy(self.data_raw)
            data_padded = np.append(data_padded, np.zeros((_m, _n)), axis=0)

            Ipad = np.average(self.data_raw[:, -2])
            Qpad = np.average(self.data_raw[:, -1])

            data_padded[-_m:, -2] = Ipad * np.ones(_m)
            data_padded[-_m:, -1] = Qpad * np.ones(_m)

            r = i1
            for ii, var_name in enumerate(self.var_names):
                base_array = np.array(self.var_vals[var_name])
                r = r / self.var_dims[var_name]
                var_vals = make_var_vals(base_array, r, i1)
                data_padded[:, ii] = var_vals

            return data_padded

    def data_xy(self, var_x, var_y, padded_data=True):
        if var_x not in self.var_names or var_y not in self.var_names:
            raise Exception('%s or %s are not valid variable names ' % (var_x,
                                                                        var_y))
        else:
            self.var_x = var_x
            self.var_y = var_y
            if padded_data:
                data_xy = self.data_padded()
            else:
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

            for ivar, var_name in enumerate(var_names_xy):
                aa = np.prod(dims_xy[ivar:])
                jj = 0
                while (jj+1)*aa <= self.num_variations_padded:
                    args = data_xy[jj*aa:jj*aa+aa, ivar].argsort()
                    data_xy[jj*aa:jj*aa+aa] = data_xy[jj*aa:jj*aa+aa][args, :]
                    jj += 1

            resh1 = copy.copy(dims_xy)
            resh1.append(2)
            resh2 = copy.copy(dims_xy)
            resh2.append(self.var_num-2)
            data_xy_reshaped = np.reshape(data_xy[:, -2:], tuple(resh1))
            var_xy_reshaped = np.reshape(data_xy[:, :-4], tuple(resh2))
            return var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped

    def plot_xy(self, var_x, var_y):
        # TODO generalize to N dimensions (not just 1+2(x,y)+2(I,Q))

        res = self.data_xy(var_x, var_y)
        var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped = res
        x = self.var_vals[var_x]
        x = np.append(x, 2*x[-1]-x[-2])
        y = self.var_vals[var_y]
        y = np.append(y, 2*y[-1]-y[-2])

        shape = np.shape(data_xy_reshaped)
        for ii in np.arange(shape[0]):
            I = data_xy_reshaped[ii, :, :, 0]
            Q = data_xy_reshaped[ii, :, :, 1]
            data_abs = np.abs(I+1j*Q)
            fig, ax = plt.subplots()
            ax.pcolor(x, y, data_abs)
            ax.set_ylabel(var_y)
            ax.set_xlabel(var_x)
            ax.set_title("%s = %s" % (var_names_xy[0],
                                      var_xy_reshaped[ii, 0, 0][0]))
            ax.axis('tight')

    def plot_xy_2(self, var_x, var_y):
        # TODO generalize to N dimensions (not just 1+2(x,y)+2(I,Q))

        res = self.data_xy(var_x, var_y)
        var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped = res
        shape = np.shape(data_xy_reshaped)
        if len(shape) > 3:
            new_shape = [np.prod(shape[:-3]), *shape[-3:]]
        else:
            new_shape = shape
        data_xy_reshaped_to_plot = np.reshape(data_xy_reshaped,
                                              tuple(new_shape))
        x = self.var_vals[var_x]
        x = np.append(x, 2*x[-1]-x[-2])
        y = self.var_vals[var_y]
        y = np.append(y, 2*y[-1]-y[-2])

        dims = []
        for var_name in var_names_xy[:-2]:
            dims.append(self.var_dims[var_name])

        for ii in np.arange(new_shape[0]):
            I = data_xy_reshaped_to_plot[ii, :, :, 0]
            Q = data_xy_reshaped_to_plot[ii, :, :, 1]
            data_abs = np.abs(I+1j*Q)
            fig, ax = plt.subplots()
            ax.pcolor(x, y, data_abs)
            ax.set_ylabel(var_y)
            ax.set_xlabel(var_x)
            title = ''
            c = 1
            for var_name in var_names_xy[:-2]:
                Nr = np.prod(dims[c:])
                Nv = self.var_dims[var_name]
                var_vals = self.var_vals[var_name]
                title += "%s = %s" % (var_name, var_vals[np.mod(int(ii/Nr),
                                                                Nv)])
                c += 1
            ax.set_title(title)
            ax.axis('tight')

    def plot_x_2(self, var_x, var_y=None):
        # TODO display variable names on each plot

        if var_y is None and self.var_num > 1:
            uu = -1
            var_y = self.var_names[uu]
            while var_y != var_x:
                uu -= 1
                var_y = self.var_names[uu]

        res = self.data_xy(var_x, var_y)
        var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped = res
        x = self.var_vals[var_x]

        shape = np.shape(data_xy_reshaped)
        if len(shape) > 2:
            new_shape = [np.prod(shape[:-2]), *shape[-2:]]
        else:
            new_shape = shape
        data_xy_reshaped_to_plot = np.reshape(data_xy_reshaped,
                                              tuple(new_shape))
        dims = []
        for var_name in var_names_xy[:-1]:
            dims.append(self.var_dims[var_name])

        for ii in np.arange(new_shape[0]):
            I = data_xy_reshaped_to_plot[ii, :, 0]
            Q = data_xy_reshaped_to_plot[ii, :, 1]
            data_abs = np.abs(I+1j*Q)
            fig, ax = plt.subplots()
            ax.plot(x, data_abs)
            ax.set_ylabel('abs')
            ax.set_xlabel(var_x)
            title = ''
            c = 1
            for var_name in var_names_xy[:-1]:
                Nr = np.prod(dims[c:])
                Nv = self.var_dims[var_name]
                var_vals = self.var_vals[var_name]
                title += "%s = %s" % (var_name, var_vals[np.mod(int(ii/Nr),
                                                                Nv)])
                c += 1
            ax.set_title(title)
            ax.axis('tight')

    def plot_x(self, var_x, var_y=None):
        # TODO display variable names on each plot

        if var_y is None and self.var_num > 1:
            uu = -1
            var_y = self.var_names[uu]
            while var_y != var_x:
                uu -= 1
                var_y = self.var_names[uu]

        res = self.data_xy(var_x, var_y)
        var_names_xy, data_xy, data_xy_reshaped, var_xy_reshaped = res
        x = self.var_vals[var_x]

        dim_params = self.num_variations_padded/len(x)

        data_xy_reshaped_to_plot = np.reshape(data_xy_reshaped,
                                              (dim_params,
                                               len(self.var_vals[var_x]), 2))

        for ii in np.arange(dim_params):
            I = data_xy_reshaped_to_plot[ii, :, 0]
            Q = data_xy_reshaped_to_plot[ii, :, 1]
            data_abs = np.abs(I+1j*Q)
            fig, ax = plt.subplots()
            ax.plot(x, data_abs)
            ax.set_ylabel(var_y)
            ax.set_xlabel(var_x)
            #ax.set_title("%s = %s" % (var_names_xy[0],
            #                          var_xy_reshaped[ii, 0, 0][0]))
            ax.axis('tight')

    def avg_along_y(self, data_xy_reshaped):
        data_avg_along_y = np.mean(data_xy_reshaped, axis=1)

        x = self.var_vals[self.var_x]
        #x = np.append(x, 2*x[-1]-x[-2])
        y = self.var_vals[self.var_y]
        #y = np.append(y, 2*y[-1]-y[-2])

        dim_params = self.num_variations_padded/len(x)/len(y)

        data_xy_reshaped_to_plot = np.reshape(data_avg_along_y,
                                              (dim_params, len(x), 2))

        for ii in np.arange(dim_params):
            I = data_xy_reshaped_to_plot[ii, :, 0]
            Q = data_xy_reshaped_to_plot[ii, :, 1]
            data_abs = np.abs(I+1j*Q)
            fig, ax = plt.subplots()
            ax.plot(x, data_abs)
            #ax.set_ylabel(var_y)
            #ax.set_xlabel(var_x)
#            ax.set_title("%s = %s" % (var_names_xy[0],
#                                      var_xy_reshaped[ii, 0, 0][0]))
            ax.axis('tight')
        return data_avg_along_y
