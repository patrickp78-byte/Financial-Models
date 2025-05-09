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

url = "G://GitHub//Financial-Models//Back Up Data//full_data_back_up_2025-05-08.pkl"

with open(url, 'rb') as r:
    loaded_data = pickle.load(r)

d_query = loaded_data["XCD.TO"]["Close"]
d_query = d_query.loc[dt.date(2020,1,1):]
d_query = d_query.reset_index().drop(columns= "Date")
d_query = d_query.pct_change().dropna().reset_index().drop(columns = "index")



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
    cond_var[0] = l_run_vol
    
    for i in range(1,len(resid)):
        cond_var[i] = omega + alpha * sqresid[i-1].item()
        
    l_likelihood = np.zeros(len(cond_var))   
    for i in range(len(cond_var)):   
        l_likelihood[i] = (-1/2) * (np.log(2*np.pi) + np.log(cond_var[i].item()) + (sqresid[i].item()/cond_var[i].item()))
    l_like = -np.sum(l_likelihood)   
    return l_like


stdev = float(d_query.std().iloc[0]) 
var= stdev**2
un_var = var
mu = float(d_query.mean().iloc[0]) 
param = [un_var,0.5]

# check = arch_model(param,d_query)

bounds = [(1e-6, None), (1e-6, 1 - 1e-6)]

constraints = [
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





