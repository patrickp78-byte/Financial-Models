# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import yfinance as yf
import pandas as pd 
import numpy as np 
import scipy 
import statsmodels
import datetime as dt
import requests
import yfinance as yf 
import os
import math
import streamlit as st
import plotly.express as px 
 

start = dt.datetime(2020,1,11)
end = dt.datetime(2024,12,1)
cad_usd = "CADUSD=X"
usd_cad = "CAD=X"
option_expiry_date = "2024-10-18"

data_to_query = ["^GSPC","^DJI","QQQM", "SFY","^GSPTSE","VDY.TO","XHU.TO",cad_usd,usd_cad]

def historical_price(data,start,end):
    hits_price = {}
    for i in data:
        query = yf.download(i, start= start, end= end)
        hits_price[i] = query.reset_index()
    return hits_price

def call_data(data,date):
    query = yf.Ticker(data)
    option = query.option_chain(date)
    call = option.calls
    return call

def put_data(data,date):
    query = yf.Ticker(data)
    option = query.option_chain(date)
    put = option.puts
    return put

query_data = historical_price(data_to_query,start,end)
call_data_hist =  call_data("QQQM","2024-10-18")

y0 = query_data["QQQM"]["Close"]
y1 = query_data["SFY"]["Close"]
# y2 = query_data["QQQM"]["VDY.TO"]

import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=query_data["QQQM"]["Date"], y=y0,
                         mode = "lines",
                         name= "QQQM"))

fig.add_trace(go.Scatter(x=query_data["SFY"]["Date"], y=y1,
                         mode = "lines",
                         name= "SFY"))

#https://plotly.com/python/line-and-scatter/

# showing the plot
show_graph = st.plotly_chart(fig)



# streamlit run "/Users/patrickpascua/Documents/GitHub/Financial-Models/streamlit Trial.py"