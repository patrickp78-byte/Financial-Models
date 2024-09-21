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


newfile_loc = "/Users/patrickpascua/Desktop/Financial Model"
os.chdir(newfile_loc)
directory_script = os.path.dirname(__file__)
print(directory_script)


t_date = dt.datetime.today() 
y_date = t_date - dt.timedelta(days = 1)
y_date = y_date.replace(hour = 0, minute= 0, second= 0, microsecond=0) ; print(y_date)
s_date = dt.datetime(2000,1,1); print(s_date)

data_to_query = ["QQQM", "QQQ"]

data_query = {}

def query(tickers_to_query):
    data = yf.download(tickers= tickers_to_query, start=s_date, end=y_date, group_by= "tickers")
    actual_data = pd.DataFrame(data["Close"])
    actual_data["Volume"] = data["Volume"]
    return actual_data

for i in data_to_query:

    data_query[i] = query(i)



print(data_query["QQQM"].head(5))
print(data_query["QQQ"].head(5))