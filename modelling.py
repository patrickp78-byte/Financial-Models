# -*- coding: utf-8 -*-
"""
Created on Sat May  3 17:18:00 2025

@author: patri
"""

import pandas as pd
import numpy as np
import datetime as dt
import pickle
import sys
import os
import scipy as sc
import arch

file_loc = os.path.abspath(__file__)
dir_path = os.path.dirname(file_loc)
os.chdir(dir_path)
print(os.getcwd())

url = "full_data.pkl"

with open(url, 'rb') as r:
    loaded_data = pickle.load(r)

d_query = loaded_data["^GSPC"]["Close"]
d_query = d_query.loc[dt.date(2010,1,1):]
d_query = d_query.reset_index().drop(columns= "Date")
d_query = d_query.pct_change().dropna().reset_index().drop(columns = "index")

# This function creates a graph from a list of a tuples
def graphing(data, t):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots() 
    for i in data:
        ax.plot(i[0], label= f'{i[1]}')  
    plt.legend()
    plt.title(t)
    plt.show()



# This function uses the arch_library to calculate the model
def arch_lib(data,mean, vol, p,o,q, res ):
    res = res.upper()
    lib_garch = arch.arch_model(data, mean =mean, vol = vol , p= p , o = o, q = q, rescale = False)
    model_garch = lib_garch.fit()
    lib_garch= model_garch.params
    libr_result = pd.DataFrame()
    libr_result['lib_resid'] = model_garch.resid
    libr_result['lib_sqr_resid'] = libr_result['lib_resid']**2
    libr_result['lib_condvol'] = model_garch.conditional_volatility
    libr_result['lib_condvar'] = libr_result['lib_condvol']**2
    if res == 'PARAMS':
        return lib_garch
    elif res == 'DF':
        return libr_result
    else:
        return []
    
#%%
##### ARCH

#### This validates the value that is obtain
lib_arch = arch.arch_model(d_query, mean ="Zero", vol = "ARCH" , p= 1 , o = 0, q = 0, rescale = False)
model = lib_arch.fit()
lib_par= model.params

### ARCH Model
def arch_model(params,data):
    #print(params)
    data = np.array(data)
    mu = 0
    omega = params[0]
    alpha = params[1]
    l_run_vol =(omega/(1-alpha))**0.5
    
    resid = data - mu
    sqresid = resid **2
    cond_var= np.zeros(len(sqresid))
    cond_var[0] = l_run_vol**2
    
    for i in range(1,len(resid)):
        cond_var[i] = omega + alpha * sqresid[i-1].item()
        
    l_likelihood = np.zeros(len(cond_var))   
    for i in range(len(cond_var)):   
        l_likelihood[i] = (-1/2) * (np.log(2*np.pi) + np.log(cond_var[i].item()) + (sqresid[i].item()/cond_var[i].item()))
    l_like = np.sum(l_likelihood)   
    return -l_like


stdev = float(d_query.std().iloc[0]) 
var= stdev**2
un_var = var
mu = float(d_query.mean().iloc[0]) 
param = [un_var,0.5]

# check = arch_model(param,d_query)

bounds = [(1e-6, None), (1e-6, 1 - 1e-6)]

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[1]},       # alpha ≥ 0
    {'type': 'ineq', 'fun': lambda x: 1 - x[1]},  # alpha < 1
    {'type': 'ineq', 'fun': lambda x: x[0]}       # omega ≥ 0
    
]

### Good MLE method 
#### "(No constraints: TNC,L-BFGS-B,Nelder-Mead ),   
### (with constraints and Bounds: trust-constr, COBYLA, COBYQA, )"

maximize = sc.optimize.minimize(fun =arch_model , x0=param, args=(d_query,),\
                                method= "L-BFGS-B" , bounds = bounds)
mle_omega,mle_alpha = maximize.x
mle_param_res = pd.DataFrame([mle_omega,mle_alpha], index = ["mle_omega","mle_alpha"])





#%%

#GARCH model
param = arch_lib(d_query, 'Zero', 'GARCH', 1,0,1,'PARAMS' )
df = arch_lib(d_query, 'Zero', 'GARCH', 1,0,1,'df' )

def garch(params, data, res):
    print(params)
    res = res.upper()
    data = np.array(data)
    mu, om, alpha,beta = 0, params[0], params[1],params[2]
    # long_run = (om/(1- alpha - beta))**0.5
    
    resid = data-mu
    sq_resid = resid**2
    
    cond_var = np.zeros(len(resid))
    cond_var[0]= om/(1- alpha - beta)
    
    for i in range(1,len(resid)):
        cond_var[i] = om + alpha * sq_resid[i-1].item() + beta* cond_var[i-1].item()
    
    log_likelihood = np.zeros(len(cond_var))
    
    for i in range(len(log_likelihood)):
        log_likelihood[i] = (-1/2) * (np.log(2*np.pi) + np.log(cond_var[i].item()) + (sq_resid[i].item()/cond_var[i].item()))
    
    sum_l_like = np.sum(log_likelihood)
    
    if res == 'CONDITIONAL_VAR':
        return cond_var
    elif res == 'CONDITIONAL_VOL':
        return cond_var**0.5
    elif res == 'RESID':
        return cond_var**0.5
    elif res == 'SQR_RESID':
        return cond_var**0.5
    elif res == 'L_LIKE':
        return -sum_l_like
    else:
        return []

stdev = float(d_query.std().iloc[0]) 
var= stdev**2
un_var = var

params = [0,0.5,0.5]
bounds = [(1e-10, None), (1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6)]

constraints = [
    {'type': 'ineq', 'fun': lambda x: x[1]},       # omega ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[2]},       # alpha ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[3]},       # beta ≥ 0
    {'type': 'ineq', 'fun': lambda x: 1 - x[2]},  # alpha < 1
    {'type': 'ineq', 'fun': lambda x: 1 - x[3]},  # beta < 1
    {'type': 'ineq', 'fun': lambda x: 1- x[2]-x[3]  } # ensures stationarity
]
        
# Uses MLE From Scipy
maximize = sc.optimize.minimize(fun =garch , x0=param, args=(d_query,'L_LIKE'),\
                                method= "L-BFGS-B" , bounds = bounds)

# These are the Parameter results from MLE
mle_omega,mle_alpha,mle_beta = maximize.x
mle_param_res = pd.DataFrame([mle_omega,mle_alpha,mle_beta], index = ["mle_omega","mle_alpha","mle_beta"])

# Graphs the results
result_mle = garch([mle_omega,mle_alpha,mle_beta], d_query, 'CONDITIONAL_Vol')
graph = [((d_query**2)**(1/2), 'actual vol'),(df['lib_condvol'], 'lib result'),(result_mle,'forcast')]
graphing(graph, "Conditional vol")

#%%

#EGARCH Model

# THis is just a test for Git LAb
    
    
