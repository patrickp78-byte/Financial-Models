# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:35:29 2025

@author: patri
"""

import pandas as pd
import numpy as np
import scipy as sc
import yfinance as yf
import datetime as dt
import os
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import requests
import sys


## This makes sure that the file is in the right directory
dir_loc = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_loc)


s_date= dt.date(2005,1,1)
e_date =dt.date.today()

ticker = ["HXQ.TO","VDY.TO","XHU.TO","ZDV.TO","ZDY.TO","ZGQ.TO","ZSP.TO","^GSPC","^NDX","^GSPTSE","^VIX","LQD","XCD.TO"]#, "CADUSD=X", "USDCAD=X", ]
#ticker = ["QQQM"]
#LQD: investment grade (US) 
#XCD.TO: investment grade (CAD)

# Market Index | Description | Ticker
# S&P 500 | Large-cap U.S. stocks | ^GSPC
# Dow Jones | 30 major U.S. companies | ^DJI
# Nasdaq 100 | Tech-heavy growth stocks | ^NDX
# Nasdaq Comp. | All Nasdaq-listed stocks | ^IXIC
# Russell 2000 | U.S. small-cap stocks | ^RUT
# VIX Index | Volatility (fear index) | ^VIX

#other_data = ["^VIX", "CADUSD=X", "USDCAD=X", "LQD","XCD.TO"]


# Market Index | Description | Ticker
# TSX Composite Index | Major Canadian market index | ^GSPTSE
# S&P/TSX 60 | 60 largest Canadian companies | ^TX60
# TSX Venture Index | Emerging companies, juniors | ^CDNX
# Canadian Dollar Index | Exchange rate: CAD to USD | CADUSD=X
# USD to CAD | Inverse exchange rate | USDCAD=X


data_pull = "Close" # This makes you pick what data you want to pull

def query(tickers,start_date,end_date,d_type):
    dataquery = yf.download(tickers,start_date,end_date, group_by= 'tickers')
    data = {}
    for i in tickers:
        data[i] = dataquery[i][d_type] 
    data = pd.DataFrame(data)
    return data

data_query = query(ticker,s_date,e_date,data_pull)
#other_query = query(other_data,s_date,e_date,data_pull)
#a_data = pd.merge(data_query, other_query, on ="Date", how = "left")
a_data = data_query.ffill()


########## #choletsky Decomp ##############
# cor =a_data.corr()
# cor = np.array(cor)
# lower_tri = np.linalg.cholesky(cor)
# lower_tri = pd.DataFrame(lower_tri, index= ticker, columns= ticker)
# cor = pd.DataFrame(cor, index= ticker, columns= ticker)



######### changes ###########
f_diff = a_data.diff().dropna()
pct_change = (a_data.pct_change()).dropna()
ln_return  = np.log(a_data/a_data.shift(1)).dropna()

###############################################
fitting_data = f_diff ##############
##############################################


############ Model
from arch import arch_model

def model(data, sym , m_type):
    g_models = arch_model(data[sym], vol = m_type, p =1 ,o=1, q=1, mean= "Constant",  rescale=False)
    result = g_models.fit(disp = "normal")
    params = result.params
    return params


p_res = {}

for i in ticker:
    model_fitting = model(fitting_data,i, "EGARCH") 
    p_res[i] = model_fitting


data_pres = pd.DataFrame(p_res)


calc = pd.DataFrame()

for i in ticker:
    calc[f'resid_{i}'] = fitting_data[i] - data_pres.loc["mu", i] 
    calc[f'c_Var_{i}'] = (fitting_data[i].std())**0.5 
    
calc = calc.reset_index()
for i in ticker:
    for n in range(len(calc)):
        calc.loc[n+1, f'c_Var_{i}']= data_pres.loc["omega", i] \
                                            + data_pres.loc["alpha[1]", i] *( abs(calc.loc[n, f'resid_{i}'] /calc.loc[n, f'c_Var_{i}']**0.5) - (2/np.pi)**2) \
                                                + data_pres.loc["gamma[1]", i] * (calc.loc[n, f'resid_{i}'] /calc.loc[n, f'c_Var_{i}']**0.5) \
                                                    + (data_pres.loc["beta[1]", i] * np.log(calc.loc[n,f'c_Var_{i}']))
        calc.loc[n+1, f'c_Var_{i}']= np.exp(calc.loc[n+1, f'c_Var_{i}'])
                                                    
calc = calc.dropna()

######### GRAPHING
def graphing(d,title):
    fig,ax = plt.subplots()
    plt.title(title)
    ax.plot(d[f'Date'], (d[f'resid_{i}']**2)**0.5)
    ax.plot(d[f'Date'], (d[f'c_Var_{i}'])**0.5)
    plt.show()
    
for i in ticker:
    graphing(calc , i)




######## Fitting Check


fit_check = pd.DataFrame()
for i in ticker:
    error = (calc[f'c_Var_{i}'][50:] - calc[f'resid_{i}'][50:])**2 
    fit_check.loc["mse",f'mse_{i}' ] = (1/len(error)) *sum(error)









sys.exit()

# Interest Rate
# US Interest Rate
us_rate = web.DataReader('FEDFUNDS', 'fred', s_date, e_date)



# Exporting
file_name = "New.xlsx"

with pd.ExcelWriter(file_name, engine= "openpyxl", mode='w') as writer:
    a_data.to_excel(writer, sheet_name = "Data", index=True)
    cor.to_excel(writer, sheet_name = "corr matrix", index=True)
    data_pres.to_excel(writer, sheet_name = "params", index=True)
    lower_tri.to_excel(writer, sheet_name = "choletsky_L", index=True)
    

print(f'Done!')