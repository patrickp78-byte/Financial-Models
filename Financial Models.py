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
import pickle


start_date = "2020-01-31"
end_date = "2023-12-31"
tickers= ["HXQ.TO", "VDY.TO", "XHU.TO", "ZSP.TO"]

data_query = yf.download(tickers = tickers, start = start_date, end= end_date, group_by= "tickers")

query = data_query.copy()

#%%

price_dict = {}

for i in tickers:
    ticker_data = query[i]
    price_dict.update({i:ticker_data.reset_index()})
    

temp_price = price_dict["HXQ.TO"]