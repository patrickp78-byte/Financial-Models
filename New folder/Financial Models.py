# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:50:53 2024

@author: patri
"""

import pandas as pd 
import  numpy as np 
import yfinance as yf 
import scipy as sp
import matplotlib.pyplot as plt
import datetime as dt 
import sys
import streamlit as st
import arch 

end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days = 90)

tickers= ["QQQM","ZDV.TO"]

data_query = yf.download(tickers = tickers, start = start_date, end= end_date)

data  = data_query.copy()


prices = data["Close"].dropna()
diff = prices["QQQM"].diff().dropna()
diff = diff.reset_index()
delta = np.array(diff["QQQM"])

ave = np.average(delta)
standard_dev = np.std(delta)**(1/2)
alpha = 0 
beta = 0

param= [ave,standard_dev,alpha,beta]
print(param)

def garch(params):
    mean,omega, alpha, beta = params[0],params[1],params[2],params[3]
    long_run = np.sqrt(omega / max((1 - alpha - beta),1e-6))
    resid = delta - mean
    resid_sqr = abs(resid)
    conditional = np.zeros(len(diff))
    conditional[0] = long_run
    for i in range(1,len(diff)):
        calc = np.sqrt(omega + alpha * resid_sqr[i - 1] + beta * conditional[i - 1] ** 2)
        conditional[i] =calc
    likelihood = 1/((2*np.pi)**(1/2)*(conditional))*np.exp(-resid_sqr**2/(2*conditional**2))
    log_like= np.sum(np.log(likelihood)+1e-10)
    return -log_like
    

data = garch(param);print(data)

bounds = [
    (1e-6, None),  # mean (unbounded)
    (1e-6, None),  # omega > 0
    (0, 1),        # 0 ≤ alpha ≤ 1
    (0, 1)         # 0 ≤ beta ≤ 1
]

constains = [{"type":"ineq","fun": lambda param: 0.999-(param[2] + param[3])}]


res = sp.optimize.minimize(garch,param,method = "SLSQP" ,
                           bounds = bounds, 
                           constraints =constains, 
                           options={"ftol": 1e-10, "maxiter": 1000} )

params = res.x
print(params)
print(-float(res.fun))


garch_model = arch.arch_model(delta, vol="Garch", p=1,q=1, mean = "Constant", dist ="normal")
garch_fit = garch_model.fit(disp = "off")
print(garch_fit.summary())

    
    