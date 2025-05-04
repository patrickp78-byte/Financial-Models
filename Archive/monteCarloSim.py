# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:31:46 2025

@author: patri
"""

import pandas as pd 
import numpy as np 
import scipy 
import statsmodels
import datetime as dt
import requests
import yfinance as yf 
import os
import math
import sys
from scipy.stats import norminvgauss
from scipy.stats import qmc
from scipy.stats import norm
import matplotlib.pyplot as plt


def monteCarloSim(column,row_num):
    rand = qmc.Sobol(d=column)# Sobol or Halton
    random_val = rand.random(row_num)
    return random_val


def norm_inverse(percentile, mean, stdev):
    r = norm.ppf(percentile, mean, stdev)
    return r

gen_rand_val = monteCarloSim(1,10000) 


result = np.array([norm_inverse(gen_rand_val[i],5,0.5) for i in range(len(gen_rand_val))])

sorted_data =np.sort(result,axis = 0)
data = pd.DataFrame(sorted_data,columns = ["gen_val"])


r1 = round((0.25/100)*len(result))
t = data["gen_val"][r1]; print(t)
r2 = round((99.75/100)*len(result))
t2 = data["gen_val"][r2]; print(t2)

filtered_data = data["gen_val"][r1:r2]
filtered_data = filtered_data.reset_index(drop = True)

fig, ax = plt.subplots()
ax.hist(data["gen_val"], bins=15)
ax.hist(filtered_data, bins=20)

plt.show(fig)





