#!/usr/bin/env python
# coding: utf-8

# Export to Python
# Required imports
import datetime
import os
import numpy as np
import pandas as pd
import gzip
import glob
import pickle
import copy
import math
from io import StringIO
import importlib.machinery
from scipy import stats, optimize
import multiprocessing
from functools import partial
import time
import pandas as pd


class LSTM:

    def __init__(self, 
                 X_size = 1,
                 H_size = 3, 
                 weight_sd = 1e-1,
                 learning_rate = 1e-1,
                 loss_option = 'mse'):
        
        self.X_size = X_size 
        self.H_size = H_size 
        self.z_size = self.H_size + self.X_size 
        self.weight_sd = weight_sd         
        self.learning_rate = learning_rate 
        self.loss_option = loss_option      
        self.w_b_indices_and_dim, self.DoF = self.get_indices_W_b()

    # Activation Functions and their derivatives
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, y):
        return 1 - y * y

    # Weights and their derivatives intialization
    def initialize_weights_and_biases(self, nb_samples):
            W, b = {}, {}
            W['f'] = np.random.randn(nb_samples, 
                                     self.H_size, 
                                     self.z_size) \
                                        * self.weight_sd + 0.5
            b['f'] = np.zeros((nb_samples, 
                               self.H_size, 
                               1))

            W['i'] = np.random.randn(nb_samples, 
                                     self.H_size, 
                                     self.z_size) \
                                        * self.weight_sd + 0.5
            b['i'] = np.zeros((nb_samples, 
                               self.H_size, 
                               1))

            W['C'] = np.random.randn(nb_samples, 
                                     self.H_size, 
                                     self.z_size) \
                                        * self.weight_sd
            b['C'] = np.zeros((nb_samples, 
                               self.H_size, 
                               1))

            W['o'] = np.random.randn(nb_samples, 
                                     self.H_size, 
                                     self.z_size) \
                                        * self.weight_sd + 0.5
            b['o'] = np.zeros((nb_samples, 
                               self.H_size, 
                               1))

            # For final layer to predict; Many to One
            W['v'] = np.random.randn(nb_samples, 
                                     1, 
                                     self.H_size) \
                                        * self.weight_sd
            b['v'] = np.zeros((nb_samples, 
                               1, 
                               1))
            
            return W, b

        
    def get_indices_W_b(self):
    
        indices_W_b = {'W': {}, 'b': {}}

        # For the f, i, C and o layers inside an LSTM unit 
        size_W_fiCo = self.H_size *  self.z_size
        size_b_fiCo = self.H_size
        # For the v layer inside an LSTM unit
        size_W_v = self.X_size * self.H_size
        size_b_v = self.X_size

        start_W = 0

        for layer_ in ['f', 'i', 'C', 'o']:

            indices_W_b['W'][layer_] = {}
            start_W, end_W = start_W, start_W + size_W_fiCo - 1
            indices_W_b['W'][layer_]['idx'] = (start_W, end_W)
            indices_W_b['W'][layer_]['dims'] = (self.H_size,  
                                                self.z_size)

            indices_W_b['b'][layer_] = {}
            start_b, end_b = end_W + 1, end_W + 1 + size_b_fiCo - 1
            indices_W_b['b'][layer_]['idx'] = (start_b, end_b)
            indices_W_b['b'][layer_]['dims'] = (self.H_size,  1)

            start_W = end_b + 1

        indices_W_b['W']['v'] = {}
        start_W, end_W = start_W, start_W + size_W_v - 1
        indices_W_b['W']['v']['idx'] = (start_W, end_W)
        indices_W_b['W']['v']['dims'] = (self.X_size, 
                                         self.H_size)

        indices_W_b['b']['v'] = {}
        start_b, end_b = end_W + 1, end_W + 1 + size_b_v - 1
        indices_W_b['b']['v']['idx'] = (start_b, end_b)
        indices_W_b['b']['v']['dims'] = (self.X_size,  1)
        
        return indices_W_b, end_W + 2
    
    def initialize_derivatives_weights_and_biases(self, W, b):

        dW, db = {}, {}

        for key_ in W.keys():

            dW[key_] = np.zeros_like(W[key_])
            db[key_] = np.zeros_like(b[key_])

        return dW, db

    def clip_derivatives(self, dW, db):

        for key_ in dW.keys():

            dW[key_] = np.clip(dW[key_], -1, 1)
            db[key_] = np.clip(db[key_], -1, 1)

        return dW, db

    def forward_one_cell_lstm(self, 
                              x, 
                              h_prev, 
                              C_prev, 
                              W, 
                              b): 
        
            z = np.hstack((h_prev, x)) 

            f = self.sigmoid(np.matmul(W['f'], z) + b['f'])
            i = self.sigmoid(np.matmul(W['i'], z) + b['i'])
            C_bar = self.tanh(np.matmul(W['C'], z) + b['C'])

            C = f * C_prev + i * C_bar
            o = self.sigmoid(np.matmul(W['o'], z) + b['o'])
            h = o * self.tanh(C)

            v = np.matmul(W['v'], h) + b['v']
            y = v # linear regressor

            return z, f, i, C_bar, C, o, h, v, y


    def forward_prop(self, 
                     inputs, 
                     h_prev, 
                     C_prev, 
                     W, 
                     b):
        
            nb_data_points = inputs.shape[1]
            z_s, f_s, i_s,  = {}, {}, {}
            C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
            v_s, y_s =  {}, {}

            # Values at t - 1
            h_s[-1] = np.copy(h_prev)
            C_s[-1] = np.copy(C_prev)

            # Loop through time steps
            for t in range(len(inputs)):
                # Time steps over the rows and different
                # data points over the columns
                x_s = inputs[t, :]            
                x_s = x_s.reshape(1,1,-1)
                # Repeat it for the nb_samples
                nb_samples = W['f'].shape[0]
                x_s = np.tile(A=x_s, reps=nb_samples).reshape(nb_samples, 
                                        self.X_size, 
                                        nb_data_points)
                
                (z_s[t], f_s[t], i_s[t],
                C_bar_s[t], C_s[t], o_s[t], h_s[t],
                v_s[t], y_s[t]) = \
                    self.forward_one_cell_lstm(x_s, h_s[t - 1], C_s[t - 1], 
                                               W = W, b = b) 

            return z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, v_s, y_s

    def loss(self, y_true, y_pred, loss_method = 'se'):
        if loss_method is "se":          
            return .5 * (y_true - y_pred)**2.
    
    def backward(self,
                 dh_next, 
                 dC_next, 
                 C_prev,
                 z, f, i, C_bar, C, o, h, 
                 W, b, dW, db, y,
                 target = None):

            if self.loss_option is 'mse':
                dv = y - target 
            elif self.loss_option is 'absent':
                dv = np.ones_like(y)

            dW['v'] += np.matmul(dv, np.transpose(h, (0,2,1)))
            db['v'] += np.sum(dv, axis=-1).                        reshape(dv.shape[0], dv.shape[1], 1) 

            dh = np.matmul(np.transpose(W['v'], (0,2,1)), dv)
            dh += dh_next

            z_transposed = np.transpose(z, (0,2,1))

            do = dh * self.tanh(C)
            do = self.dsigmoid(o) * do
            dW['o'] += np.matmul(do, z_transposed)
            db['o'] += np.sum(do, axis=-1)                        .reshape(do.shape[0], do.shape[1], 1) 

            dC = np.copy(dC_next)
            dC += dh * o * self.dtanh(self.tanh(C))
            dC_bar = dC * i
            dC_bar = self.dtanh(C_bar) * dC_bar
            dW['C'] += np.matmul(dC_bar, z_transposed)
            db['C'] += np.sum(dC_bar, axis=-1)                        .reshape(dC_bar.shape[0], dC_bar.shape[1], 1) 

            di = dC * C_bar
            di = self.dsigmoid(i) * di
            dW['i'] += np.matmul(di, z_transposed)
            db['i'] += np.sum(di, axis=-1)                        .reshape(di.shape[0], di.shape[1], 1)

            df = dC * C_prev
            df = self.dsigmoid(f) * df
            dW['f'] += np.matmul(df, z_transposed)
            db['f'] += np.sum(df, axis=-1)                        .reshape(df.shape[0], df.shape[1], 1)

            dz = (np.matmul(np.transpose(W['f'], (0,2,1)), df)
                 + np.matmul(np.transpose(W['i'], (0,2,1)), di)
                 + np.matmul(np.transpose(W['C'], (0,2,1)), dC_bar)
                 + np.matmul(np.transpose(W['o'], (0,2,1)), do))
            dh_prev = dz[:, :self.H_size, :] 
            dC_prev = f * dC

            return dW, db, dh_prev, dC_prev

    def backward_hessian(self,
                         dh_next, 
                         dC_next, 
                         C_prev,
                         z, f, i, C_bar, C, o, h, 
                         W, b, hessian, y):

            update_dW, update_db = {}, {}
            
            dv = np.ones_like(y) 

            update_dW['v'] = np.matmul(dv, np.transpose(h, (0,2,1)))
            update_db['v'] = np.sum(dv, axis=-1).                                reshape(dv.shape[0], dv.shape[1], 1)

            dh = np.matmul(np.transpose(W['v'], (0,2,1)), dv)
            dh += dh_next

            z_transposed = np.transpose(z, (0,2,1))

            do = dh * self.tanh(C)
            do = self.dsigmoid(o) * do
            update_dW['o'] = np.matmul(do, z_transposed)
            update_db['o'] = np.sum(do, axis=-1)                                .reshape(do.shape[0], do.shape[1], 1) 

            dC = np.copy(dC_next)
            dC += dh * o * self.dtanh(self.tanh(C))
            dC_bar = dC * i
            dC_bar = self.dtanh(C_bar) * dC_bar
            update_dW['C'] = np.matmul(dC_bar, z_transposed)
            update_db['C'] = np.sum(dC_bar, axis=-1)                                .reshape(dC_bar.shape[0], dC_bar.shape[1], 1) 

            di = dC * C_bar
            di = self.dsigmoid(i) * di
            update_dW['i'] = np.matmul(di, z_transposed)
            update_db['i'] = np.sum(di, axis=-1)                        .reshape(di.shape[0], di.shape[1], 1) 

            df = dC * C_prev
            df = self.dsigmoid(f) * df
            update_dW['f'] = np.matmul(df, z_transposed)
            update_db['f'] = np.sum(df, axis=-1)                                .reshape(df.shape[0], df.shape[1], 1)

            dz = (np.matmul(np.transpose(W['f'], (0,2,1)), df)
                 + np.matmul(np.transpose(W['i'], (0,2,1)), di)
                 + np.matmul(np.transpose(W['C'], (0,2,1)), dC_bar)
                 + np.matmul(np.transpose(W['o'], (0,2,1)), do))
            dh_prev = dz[:, :self.H_size, :] # ??? CHECK
            dC_prev = f * dC

            update_dW_db_ar = self.concatenate_weights_and_biases_all_layers(update_dW, 
                                                                             update_db)          
            
            update_hessian = np.einsum("...i,...j", update_dW_db_ar, update_dW_db_ar)
                        
            hessian += update_hessian
            
            return hessian, dh_prev, dC_prev
        
    def backward_prop(self, 
                      inputs,                        
                      W, b, 
                      z_s, f_s, i_s, 
                      C_bar_s, C_s, 
                      o_s, h_s, y_s,
                      targets = None):

        dW, db = self.initialize_derivatives_weights_and_biases(W, b)

        dh_next = np.zeros_like(h_s[0]) 
        dC_next = np.zeros_like(C_s[0]) 
       
        for t in reversed(range(len(inputs))):
            # Backward pass
            dW, db, dh_next, dC_next =                 self.backward(dh_next = dh_next,
                         dC_next = dC_next, C_prev = C_s[t-1],
                         z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
                         C = C_s[t], o = o_s[t], h = h_s[t], 
                         W = W, b = b, dW = dW, db = db,
                              y = y_s[t],
                         target = targets[t, :])
            
        dW, db = self.clip_derivatives(dW, db)
        
        # Hessian
        nb_samples = W['f'].shape[0]
        hessian = np.zeros((nb_samples, self.DoF, self.DoF))
        
        dh_next = np.zeros_like(h_s[0]) 
        dC_next = np.zeros_like(C_s[0]) 
        for t in reversed(range(len(inputs))):
            # Backward pass
            hessian, dh_next, dC_next =                 self.backward_hessian(dh_next = dh_next,
                         dC_next = dC_next, C_prev = C_s[t-1],
                         z = z_s[t], f = f_s[t], i = i_s[t], C_bar = C_bar_s[t],
                         C = C_s[t], o = o_s[t], h = h_s[t], 
                         W = W, b = b, hessian = hessian, 
                              y = y_s[t])        
            
        return dW, db, hessian

    def update_param_by_grad_descent(self,
                                     param, 
                                     deriv_param, 
                                     learning_rate = .01):   
        
        return param - learning_rate * deriv_param

    def update_weights_biases_for_all_layers(self,
                                             W, b, 
                                             deriv_W, 
                                             deriv_b, 
                                             learning_rate = .01):

        for layer_ in W.keys():
            W[layer_] = self.update_param_by_grad_descent(W[layer_], deriv_W[layer_], learning_rate)
            b[layer_] = self.update_param_by_grad_descent(b[layer_], deriv_b[layer_], learning_rate)

        return W, b
    
    def concatenate_two_arrays(self, 
                               main_array, 
                               second_array):

        second_array_flattened = second_array.reshape(-1)
        main_array = np.concatenate([main_array, second_array_flattened])

        return main_array


    def concatenate_weights_and_biases_all_layers(self, W, b):

        w_and_b_all_layers_concat = []
            
        for sample_ in range(W['f'].shape[0]):

            w_and_b_all_layers_concat_temp = []

            for layer_ in ['f', 'i', 'C', 'o', 'v']:

                w_and_b_all_layers_concat_temp =                         self.concatenate_two_arrays(w_and_b_all_layers_concat_temp, 
                                               W[layer_][sample_])

                w_and_b_all_layers_concat_temp =                         self.concatenate_two_arrays(w_and_b_all_layers_concat_temp, 
                                               b[layer_][sample_])

            if sample_ == 0:

                w_and_b_all_layers_concat = w_and_b_all_layers_concat_temp

            else:

                w_and_b_all_layers_concat = np.vstack((w_and_b_all_layers_concat, 
                                                      w_and_b_all_layers_concat_temp))

        if len(w_and_b_all_layers_concat.shape) == 1:
            w_and_b_all_layers_concat = w_and_b_all_layers_concat.reshape(1, -1)
            
            
        return w_and_b_all_layers_concat
    
    def extract_array_with_original_shape_from_concat_arrays(self,
                                                             concatenated_array,
                                                             indices_spec_array,
                                                             dims_specific_array):

        return concatenated_array[indices_spec_array[0]: indices_spec_array[1] + 1].                    reshape(dims_specific_array)
    
    def extract_weights_biases_all_layers_all_samples_from_concat(self, w_b_params):

        W, b = {}, {}

        for sample_ in range(w_b_params.shape[0]):

            for key_ in self.w_b_indices_and_dim['W'].keys():
                temp = self.extract_array_with_original_shape_from_concat_arrays(w_b_params[sample_],
                                                                            self.w_b_indices_and_dim['W'][key_]['idx'],
                                                                            self.w_b_indices_and_dim['W'][key_]['dims'])
                temp = temp.reshape(1, temp.shape[0], temp.shape[1])
                if sample_ == 0:
                    W[key_] = temp
                else:
                    W[key_] = np.vstack((W[key_], temp))

        for sample_ in range(w_b_params.shape[0]):

            for key_ in self.w_b_indices_and_dim['b'].keys():
                temp = self.extract_array_with_original_shape_from_concat_arrays(w_b_params[sample_],
                                                                            self.w_b_indices_and_dim['b'][key_]['idx'],
                                                                            self.w_b_indices_and_dim['b'][key_]['dims'])
                temp = temp.reshape(1, temp.shape[0], temp.shape[1])
                if sample_ == 0:
                    b[key_] = temp
                else:
                    b[key_] = np.vstack((b[key_], temp))
                    
        if len(W['f'].shape) == 2:
            for key_ in W.keys():
                W[key_] = W[key_].reshape(1, 
                                          W[key_].shape[0], 
                                          W[key_].shape[1])
                b[key_] = b[key_].reshape(1, 
                                          b[key_].shape[0], 
                                          b[key_].shape[1])

        return W, b
    
    def forward_solve(self, X, w_b_params):
        W, b = self.extract_weights_biases_all_layers_all_samples_from_concat(w_b_params)   

        nb_samples = w_b_params.shape[0]
        nb_data_points = X.shape[0]
        h_prev = np.zeros((nb_samples, 
                           self.H_size, 
                           nb_data_points))
        C_prev = np.zeros((nb_samples, 
                           self.H_size, 
                           nb_data_points)) 

        self.z_s, self.f_s, self.i_s, self.C_bar_s,         self.C_s, self.o_s, self.h_s, self.v_s, self.y_s =             self.forward_prop(inputs = X.T, h_prev = h_prev, 
                              C_prev = C_prev, W = W, b = b)

        y_s_as_array = pd.DataFrame(list(self.y_s.items()))[1].apply(lambda x: x.reshape(-1)).values
        y_s_as_array = np.stack(y_s_as_array)
        return y_s_as_array


    def jacobian_forward_solve(self, X, w_b_params, Y= None):
          
        W, b = self.extract_weights_biases_all_layers_all_samples_from_concat(w_b_params)

        if Y is None:

            targets = None

        else:

            targets = Y.T

        dW, db, hessian = self.backward_prop(inputs = X.T, 
                                            targets = targets, 
                                            W = W, b = b, 
                                            z_s = self.z_s, 
                                            f_s = self.f_s, 
                                            i_s = self.i_s, 
                                            C_bar_s = self.C_bar_s, 
                                            C_s = self.C_s, 
                                            o_s = self.o_s, 
                                            h_s = self.h_s,
                                            y_s = self.y_s)

        return self.concatenate_weights_and_biases_all_layers(dW, db), hessian


# In[ ]:


class BLSTM(LSTM):
    def __init__(self):
        super().__init__()
        
        # Prior setup
        self.mu0 = np.zeros((self.DoF, 1))
        self.std0 = np.ones((self.DoF,1))
        self.var0 = self.std0 ** 2
        self.Prec0 = np.eye(self.DoF)[:,:,np.newaxis] / self.var0
        
        # Noise setup
        self.stdn = 0.1
        self.varn = self.stdn ** 2
        
    def getMinusLogPrior(self, thetas):
        return 0.5 * np.sum( ( thetas - self.mu0 ) ** 2 / self.var0, 0 )
    
    def getGradientMinusLogPrior(self, thetas):
        return  (thetas - self.mu0) / self.var0
    
    def getForwardModel(self, thetas, X):       
        return self.forward_solve(X, thetas.T)
    
    def getMinusLogPredLikelihood(self, thetas, X, y, *arg):
        F = arg[0][np.newaxis, -1,:] if len(arg) > 0             else self.getForwardModel(thetas, X)[np.newaxis, -1,:] 
        return 0.5 * (F - y) ** 2 / self.varn
    
    def getMinusLogLikelihood(self, thetas, Y):
        X = np.hstack( (np.zeros( (1,1) ), Y[:,:-1]) )
        F = self.getForwardModel(thetas, X)
        mllkd = 0.5 * np.sum( (Y.T - F) ** 2, 0 ) / self.varn
        gmllkd, Hmllkd = self.jacobian_forward_solve(X, thetas.T, Y)
        gmllkd /= self.varn
        Hmllkd /= self.varn
        return mllkd, gmllkd.T, np.moveaxis(Hmllkd,0,2)
    
    def getMinusLogPosterior(self, thetas, Y):
        mllkd, gmllkd, Hmllkd = self.getMinusLogLikelihood(thetas, Y)
        mlpt = self.getMinusLogPrior(thetas) + mllkd
        gmlpt = self.getGradientMinusLogPrior(thetas) + gmllkd
        Hmlpt = self.Prec0 + Hmllkd
        return mlpt, gmlpt, Hmlpt
    
    def func4MAP(self, thetas, Y):
        F = self.getForwardModel(thetas, Y)
        mllkd = 0.5 * np.sum( (Y.T - F) ** 2, 0 ) / self.varn
        return (self.getMinusLogPrior(thetas) + mllkd).squeeze()
    
    def getMAP(self, Y, *arg):
        if len(arg) > 0:
            x0 = arg[0] 
        else:
            W, b = self.initialize_weights_and_biases(1)    
            x0 = self.concatenate_weights_and_biases_all_layers(W, b)
        
        res = optimize.minimize(lambda theta: self.func4MAP(theta.reshape(self.DoF, 1), Y = Y), x0, method = 'L-BFGS-B')
        return res.x.reshape(self.DoF, 1)


# In[ ]:


class SMC:
    def __init__(self, model, Y):
        self.model = model
        self.DoF   = model.DoF       
        self.nParticles = 100            
                        
    def apply(self, Y):
        self.importanceStatistics(Y)
        self.particles = self.getImportanceSamples()
        self.weights   = self.getImportanceWeights(Y)
        if self.isResamplingRequired() == True:
            self.resampleParticles()
        
    def importanceStatistics(self, Y):
        self.MAP = self.model.getMAP(Y = Y)
        mlpt, gmlpt, self.H = self.model.getMinusLogPosterior(self.MAP, Y)
        self.H = self.H.squeeze()
        self.C   = np.linalg.inv(self.H)
            
    def getImportanceSamples(self):
        return np.random.multivariate_normal(np.ndarray.flatten(self.MAP), self.C, self.nParticles).T
        
    def getImportanceWeights(self, Y):
        self.mlpt = self.model.func4MAP(self.particles, Y)
        alpha = np.exp( - self.mlpt + self.getMinusLogImportanceDensity(self.particles) ) 
        return alpha / np.sum(alpha)
        
    def resampleParticles(self):
        U = ( np.random.uniform() + np.arange(self.nParticles) ) / self.nParticles
        cumWeights = np.cumsum(self.weights)
        N_i = np.zeros(self.nParticles, 'i')
        i, j = 0, 0
        while i < self.nParticles:
            if U[i] < cumWeights[j]:
                N_i[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[:,N_i]
        self.weights   = np.ones(self.nParticles) / self.nParticles
        
    def getMinusLogImportanceDensity(self, thetas):
        shift = thetas - self.MAP;
        return 0.5 * np.sum(shift * np.matmul(self.H, shift), 0) 
           
    def isResamplingRequired(self):
        return 2 < self.nParticles * np.sum(self.weights ** 2) 


# In[ ]:


class BOCPD(BLSTM):
    def __init__(self, data):
        super().__init__()
        self.model = BLSTM()
        
        # Import data
        self.data = data
        self.nData = len(self.data)
        
        # Setup computational effort
        self.rmax = 30     # Max run length 
        self.npts = 100    # Number of posterior samples
        self.nrls = 100    # Number of run length samples
        
        # Setup credible interval
        self.flag_lci = 1      # Test left tail -> fire up drastic drecrease
        self.flag_rci = 1      # Test right tail -> fire up drastic increase
        self.risk_level_l = 2.5    # Left percentage of probability risk     
        self.risk_level_r = 2.5    # Right percentage of probability risk
        self.pred_mean = np.zeros(self.nData)                         # Predictive mean
        if self.flag_lci: self.percentile_l = np.zeros(self.nData)    # Left percentile (if flagged up)
        if self.flag_rci: self.percentile_r = np.zeros(self.nData)    # Right percentile (if flagged up)
                    
        # Changepoint prior
        self.rlr = 1000             # Run length rate hyper-parameter -> decrease to weigh changepoint prob more 
        self.H   = 1 / self.rlr     # Hazard rate
        
        # Initialize run length probabilities
        self.jp  = 1          # Joint 
        self.rlp = self.jp    # Posterior
        
        # Initialize posterior samples and probabilities
        self.pts = {}             
        self.pts[0] = self.mu0 + self.std0 * np.random.normal( size = (self.DoF, self.npts) )       
        self.ptp = {}             
        self.ptp[0] = np.exp( - 0.5 * np.sum( ( self.pts[0] - self.mu0 ) ** 2 / self.var0, 0 ) )
        self.ptp[0] /= np.sum(self.ptp[0])
        
        # Risk tolerance
        self.riskTol = 0
        
        # Initialize changepoints
        self.changepoints = {}
          
    def apply(self):            
        for t_ in range(self.nData):
            print('Time:', t_)
            
            # Sample from posterior run length
            self.getRunLengthSamples(t_)      
            
            # Update models and samplers
            self.data2t = self.data[:t_]
            self.updateInferenceModels(t_)
            
            # Get predictive samples
            self.getPredictiveSamples(t_)
            
            # Get predictive statistics
            self.getPredictiveStats(t_)
            
            # Observe new data point
            datat = self.data[t_]
            
            # Check whether changepoint            
            self.checkIfChangepoint(t_, datat)
            
            # Get run length posterior           
            self.getRunLengthProbability(t_, datat)
       
    def getRunLengthSamples(self, t_):
        if t_ != 0:
            rld = stats.rv_discrete( values = ( np.arange( min(t_+1, self.rmax) ), self.rlp ) )
            self.rls = rld.rvs( size = self.nrls )
        else:
            self.rls = np.zeros(self.nrls)
            
    def updateInferenceModels(self, t_):
        self.trmax = min(t_ + 1, self.rmax)
        if t_ != 0:
            results = pool.map( self.updateInferenceModels2Pool, range(1, self.trmax) )
            for r_ in range(1, self.trmax):
                self.pts[r_]   = results[r_-1][0]  
                Y = self.data2t[-r_:].reshape(1,r_)
                self.ptp[r_]   = 1 / (1 + results[r_-1][1])
                sumptp = np.sum( self.ptp[r_] )
                if sumptp < 1e-20: 
                    self.ptp[r_] = np.ones(self.npts) / self.npts
                else:
                    self.ptp[r_] /= sumptp
                
    def updateInferenceModels2Pool(self, r_):
        model = BLSTM()
        Y = self.data2t[-r_:].reshape(1,r_)
        smc = SMC(model, Y)
        smc.apply(Y)
        return (smc.particles, smc.mlpt)
    
    def getPredictiveSamples(self, t_):
        self.pps = np.zeros( (1,0) )  
        r_vals, r_counts = np.unique(self.rls, return_counts = 1) 
        for k_ in range( len(r_vals) ): # PARALLELIZABLE!
            r_val = r_vals[k_]
            r_count = r_counts[k_]
            
            ptd = stats.rv_discrete( values = ( np.arange(self.npts), self.ptp[r_val] ) )
            thetas = self.pts[r_val][:,ptd.rvs(size = r_count)]
            X = np.zeros( (1,1) )
            if r_val > 0: 
                X = np.hstack( (X, self.data2t[-r_val:].reshape(1,r_val)) )
            F = self.model.getForwardModel(thetas, X)[-1, :]
            tmp = F + self.stdn * np.random.normal( size = (1, r_count) )
            self.pps = np.hstack( ( self.pps, tmp ) )
       
    def getPredictiveStats(self, t_):   
        self.pred_mean[t_] = np.mean(self.pps)
        
        if self.flag_lci == 1:
            self.percentile_l[t_] = np.percentile(self.pps, self.risk_level_l)        
        if self.flag_rci == 1:
            self.percentile_r[t_] = np.percentile(self.pps, 100 - self.risk_level_r)
                       
    def getRunLengthProbability(self, t_, datat):   
        pp = np.zeros(self.trmax)
        for r_ in range(len(pp)): # PARALLELIZABLE!
            X = np.zeros( (1,1) )    
            if r_ > 0: X = np.hstack( (X, self.data2t[-r_:].reshape(1,r_)) )
            pp[r_] = np.mean( np.exp( - self.model.getMinusLogPredLikelihood( self.pts[r_], X, datat ) ) )
          
        # Calculate run length posterior
        jppp = self.jp * pp                            # Joint x predictive prob
        gp = jppp * ( 1 - self.H )                     # Growth prob
        cp = np.sum( jppp * self.H )                   # Changepoint prob
        self.jp  = np.hstack( (cp, gp) )[:self.rmax]   # Joint prob
        ep = np.sum( self.jp )                         # Evidence prob
        if ep < 1e-20:
            self.jp = np.ones(len(self.jp)) / len(self.jp)
            ep = 1.0
        self.rlp = self.jp / ep                        # Run length posterior
              
    def checkIfChangepoint(self, t_, datat):
        if self.flag_lci:           
            if datat < self.percentile_l[t_]: 
                risk = np.abs( ( datat - self.percentile_l[t_] ) / ( self.pred_mean[t_] - self.percentile_l[t_] ) )
                if risk > self.riskTol:
                    self.changepoints.update( {t_: risk} )   
                    print('Changepoint at time', t_, 'for drastic decrease')
                
        if self.flag_rci: 
            if datat > self.percentile_r[t_]: 
                risk = np.abs( ( datat - self.percentile_r[t_] ) / ( self.pred_mean[t_] - self.percentile_r[t_] ) )
                if risk > self.riskTol:
                    self.changepoints.update( {t_: risk} )   
                    print('Changepoint at time', t_, 'for drastic increase')


# In[22]:

globs=glob.glob('*.csv')
print(globs)
for g in globs:
    fname=g.split('/')[-1].split('.')[0]
    print(fname)
    df=pd.read_csv(g,header=0,index_col=0)
    for ind in range(128):
        pool=multiprocessing.Pool(40)
        temp_data=df.values[:,ind].reshape(df.values.shape[0],1)

        bocpd=BOCPD(temp_data)
        bocpd.apply()
        cpts=bocpd.changepoints
        save_fname=fname+'_'+ind+'.txt'
        numpy.savetxt(save_fname,cpts) 
        pool.close()

