# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:01:36 2025

@author: patrick
"""
import pickle
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import sys
import os
import scipy as sc
import arch



#opening pickle file
def opening_pickle(file_name):
    with open (f'{file_name}', 'rb') as f:
        query_data = pickle.load(f)
    return query_data

# Turning pickle file into Dataframe
def dict_to_dataframe(symbol, pickle_data, data_type):
    data_range = pd.date_range(start="1980-01-01", end = dt.date.today(), freq="D")
    df = pd.DataFrame(data_range).rename(columns ={0:"Date"})
    
    for i in symbol:
        pars_data = pickle_data[f'{i}'][f"{data_type}"].reset_index()
        df = pd.merge(df, pars_data,on = "Date", how ="outer")
    return df


# Kernel Density Estimator
def kde(data,smoothness = None):
    
    #https://en.wikipedia.org/wiki/Kernel_density_estimation
    
    """ This function performs Kernel Density Estimation (KDE) using a Gaussian kernel (NORMAL DENSITY). 
    It first sorts and cleans the input data, then estimates the data's probability density function (PDF) 
    at each point. If no smoothness (bandwidth) is provided, it uses Scott’s Rule, which adapts to the 
    data's spread using the minimum of standard deviation and scaled IQR.
    
    """
        
    sort = data.sort_values(ascending = True).reset_index().drop(columns = "index").dropna()
    sort = np.array(sort)
    n = len(sort)
    kernel_density = np.zeros(len(sort))
    
    if smoothness is None:
        # Scott's Rule 
        std = np.std(data, ddof = 1)
        # Calculating IQR
        q75, q25 = np.percentile(sort, 75), np.percentile(sort, 25)
        iqr = q75 - q25
        smoothness = 0.9 * min(std,iqr/1.34) * (n **(1/5))
        
        # Another Way to calculate smoothness is Silverman's Rule: smoothness = std * (4/(3*n))**(1/5)
    else:
        smoothness = smoothness
    
    for i in range(len(kernel_density)):
        calc = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((sort[i]- sort)/smoothness)**2)
        kernel_density[i] = np.sum(calc)
    kernel_density= 1/(smoothness*n) * kernel_density
    x,y = sort,kernel_density
    return x,y







# This function uses the arch_library to calculate the model
def arch_lib(data,mean, vol, p,o,q,power, res ):
    
    """ 
    The arch_lib function estimates a GARCH-type volatility model using the arch library. 
    It fits the model to the provided data using specified parameters for the mean model, 
    volatility process (GARCH, GJR-GARCH, etc.), and lag orders (p, o, q).
    After fitting, it either returns the model parameters, a DataFrame containing residuals 
    and conditional variances, or an empty list depending on the res argument.
    """
    
    res = res.upper()
    
    if power is None:
    
        lib_garch = arch.arch_model(data, mean =mean, vol = vol , p= p , o = o, q = q, rescale = False)
    else:
        lib_garch = arch.arch_model(data, mean =mean, vol = vol , p= p , o = o, q = q,power = power, rescale = False)
    
    model_garch = lib_garch.fit()
    lib_garch= model_garch.params
    libr_result = pd.DataFrame()
    libr_result['lib_resid'] = model_garch.resid
    libr_result['lib_sqr_resid'] = libr_result['lib_resid']**2
    libr_result['lib_condvol'] = model_garch.conditional_volatility
    libr_result['lib_condvar'] = libr_result['lib_condvol']**2
    libr_result['lib_loglikelihood'] =  model_garch.loglikelihood
    if res == 'PARAMS':
        return lib_garch
    elif res == 'DF':
        return libr_result
    else:
        return print(f'Select Params or DF')


# This function calculates EGARCH(1,0,1)
def egarch(params,data,t): 
    """ 
    The egrach function computes conditional variances using the EGARCH(1,1) model 
    and optionally returns either the variances or the negative log-likelihood for parameter estimation. 
    It uses the input residuals (data["diff"]) and EGARCH parameters (omega, alpha, beta, long_runvol) 
    to model time-varying volatility.
    """ 
    print(params)
    omega, alpha,beta,long_runvol = params[0],params[1],params[2],params[2]
    
    resid = data
    sq_resid = resid **2
    conditional_var = np.zeros(len(resid))
    
    conditional_var[0] = long_runvol
    for i in range(1,len(conditional_var)):
        conditional_var[i] =np.exp( omega + alpha *(np.abs(resid.loc[i-1,"diff"]/np.sqrt(conditional_var[i-1])) - np.sqrt(2/np.pi)) + beta* np.log(conditional_var[i-1]))
    l_likelhood = np.zeros(len(conditional_var))
    for i in range(len(l_likelhood)):
        l_likelhood[i] = (-1/2) * (np.log((conditional_var[i]**2) * 2 * np.pi) +(sq_resid.loc[i,"diff"]/(conditional_var[i]**2) ))
    
    l_sum = np.sum(l_likelhood)
    
    if t == "CVOL":
        return conditional_var **(1/2)
    else:
        return -l_sum

# params = [0,0.5,0.5,0.5]
# bounds = [(1e-10, None), (1e-6, 1), (1e-6, 1 ),(1e-10, None)]
# constraints = [
#     {'type': 'ineq', 'fun': lambda x: x[1]},       # omega ≥ 0
#     {'type': 'ineq', 'fun': lambda x: x[2]},       # alpha ≥ 0
#     {'type': 'ineq', 'fun': lambda x: x[3]},       # beta ≥ 0
#     {'type': 'ineq', 'fun': lambda x: 1 - x[2]},  # alpha < 1
#     {'type': 'ineq', 'fun': lambda x: 1 - x[3]},  # beta < 1
#     {'type': 'ineq', 'fun': lambda x: 1- x[2]-x[3]  } # ensures stationarity
# ]
# maximize = sc.optimize.minimize(fun =egrach , x0=params, args=(data,None),\
#                                 method= "trust-constr" , bounds = bounds)    
# mle_omega,mle_alpha,mle_beta,mle_longrun = maximize.x
# mle_param_res = pd.DataFrame([mle_omega,mle_alpha,mle_beta,mle_longrun], index = ["mle_omega","mle_alpha","mle_beta","mle_longrun"])
    


# This function calculates APARCH(1,0,1)
def aparch(params,data,t):
    """
    The aparch function estimates conditional volatility using a Power-GARCH model without leverage. 
    It takes model parameters, residual data, and a mode flag. If the flag is "C_VOL", 
    it returns the conditional volatility series; otherwise, it returns the negative log-likelihood for 
    use in parameter estimation. The model uses a power term 𝛿 to generalize the volatility dynamics.
    
    Power-GARCH is a flexible volatility model that generalizes GARCH by raising the standard deviation 
    to a power δ. This allows it to better capture features like heavy tails and nonlinear volatility. 
    It includes standard GARCH as a special case when 𝛿=2
    
    """
    
    #https://vlab.stern.nyu.edu/docs/volatility/APARCH
    print(params)
    # t = t.upper()
    
    omega,alpha,beta,delta,long_run = params[0],params[1],params[2],params[3],params[4]
    resid = np.array(data)
    abs_resid = np.abs(resid)
    conditional_vol = np.zeros(len(resid))
    conditional_vol[0] = long_run
    for i in range(1,len(conditional_vol)):
        conditional_vol[i] = (omega + alpha*(abs_resid[i-1].item()**delta )+ beta*(conditional_vol[i-1].item()**(delta)))**(1/delta)
        
    l_like = np.zeros(len(conditional_vol))
    
    for i in range(len(l_like)):
        l_like[i] = (-1/2) * (np.log((conditional_vol[i].item()**2) * 2 * np.pi) +(resid[i].item()**2/conditional_vol[i].item()**2 ))
    s_likelihood = np.sum(l_like)
    
    if t == "CVOL":
        return conditional_vol 
    else:
        return -s_likelihood
    
# params =  [0.01, 0.1, 0.8, 1.0, 1.0]
# bounds = [(1e-10, None), (1e-6, 1), (1e-6, 1 ),(1e-10, None),(1e-10, None)]
# constraints = [
#     {'type': 'ineq', 'fun': lambda x: x[1]},       # omega ≥ 0
#     {'type': 'ineq', 'fun': lambda x: x[2]},       # alpha ≥ 0
#     {'type': 'ineq', 'fun': lambda x: x[3]},       # beta ≥ 0
#     {'type': 'ineq', 'fun': lambda x: 1 - x[2]},  # alpha < 1
#     {'type': 'ineq', 'fun': lambda x: 1 - x[3]},  # beta < 1
#     {'type': 'ineq', 'fun': lambda x: 1- x[2]-x[3]  } # ensures stationarity
# ]
# maximize = sc.optimize.minimize(fun =aparch , x0=params, args=(data,None),\
#                                 method= "L-BFGS-B" , bounds = bounds)
# mle_omega,mle_alpha,mle_beta,mle_delta,mle_longrun = maximize.x
# mle_param_res = pd.DataFrame([mle_omega,mle_alpha,mle_beta,mle_delta,mle_longrun], index = ["mle_omega","mle_alpha","mle_beta","mle_delta","mle_longrun"])
# check = aparch([mle_omega,mle_alpha,mle_beta,mle_delta,mle_longrun ] ,data,t = "C_VOL" )