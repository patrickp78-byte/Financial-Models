# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


# newfile_loc = "/Users/patrickpascua/Desktop/Financial Model"
# os.chdir(__file__)
directory_script = os.path.dirname(__file__)


t_date = dt.datetime.today() 
y_date = t_date - dt.timedelta(days = 1)
y_date = y_date.replace(hour = 0, minute= 0, second= 0, microsecond=0) ; print(y_date)
s_date = dt.datetime(2000,1,1) #; print(s_date)


cad_usd = "CADUSD=X"
usd_cad = "CAD=X"

data_to_query = ["^GSPC","^DJI","QQQM", "SFY","^GSPTSE","VDY.TO","XHU.TO",cad_usd,usd_cad]

data_query = {}

def query(tickers_to_query):
    data = yf.download(tickers= tickers_to_query, start=s_date, end=y_date, group_by= "tickers")
    actual_data = pd.DataFrame(data["Close"])
    # actual_data["Volume"] = data["Volume"]
    return actual_data

for i in data_to_query:
    data_query[i] = query(i)

df_query = pd.DataFrame()

for i in range(len(data_to_query)):
    df_query[data_to_query[i]] = data_query[data_to_query[i]]

# saving_as_csv = df_query.to_csv("Data.csv") # This turns the data into excel format


df_query = df_query.reset_index()
parse_data = df_query.loc[df_query["Date"] > dt.datetime(2021,1,1)]
parse_data = parse_data.reset_index().drop(columns = "index").set_index("Date")


first_diff = parse_data.diff().dropna()
# first_diff = first_diff.dropna()

correlation_matrix = first_diff.corr()


def normal_dist(param, data):
    
    # This is the OU Model
    mu, sigma = param
    # norm = []
    # # data= data.reset_index()
    
    # for i in range(len(data[0])):
    #     norm_dist = theta*(data[0][i]) + error
    #     norm.append(norm_dist)
        
        

    # This is the 
    # prob_density = np.exp(-0.5*((np.array(data)/sigma)**2) / (math.sqrt(2) * sigma))
    # prob_density = np.log(prob_density)
    
    temp=[]
    
    for i in data:
        prob_density = np.exp(((i- mu)**2)/(2 * (sigma**2))) 
        prob_density = prob_density/ (math.sqrt(2*math.pi) * sigma)
        prob_density = np.log(prob_density)
        temp.append(prob_density)
    return -np.sum(temp)

paramiters = [0,1]

# result = normal_dist(paramiters, first_diff[usd_cad])



# data_to_calibrate =  first_diff["^GSPC"]



opt_val = scipy.optimize.minimize(normal_dist, paramiters, bounds=[(-10,5),(0.0001,5)], args = (first_diff["XHU.TO"]), method= "powell")




# mean = 0
# sd = 1
# x = 1
# result = normal_dist(x, mean, sd)

