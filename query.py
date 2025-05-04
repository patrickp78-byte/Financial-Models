# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:14:28 2025

@author: patri
"""
# This are all the tickers that is being queried

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pickle
import sys
import os


#Current working dir
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
print(f'old_dir: {script_dir}')
#changing
os.chdir(script_dir)
#new directory
print(f'new_dir: { os.getcwd() }')




# Data query
tickers1 = ["HXQ.TO","VDY.TO","XHU.TO","ZDV.TO","ZDY.TO","ZGQ.TO","ZSP.TO",\
             "^GSPC", "^DJI", "^NDX","^IXIC","^RUT","^VIX",\
                "^GSPTSE","^CDNX", \
                    "LQD","XCD.TO", \
           "CADUSD=X", "USDCAD=X", \
         ]

bond_etfs = [
    "TLT",   # iShares 20+ Year Treasury Bond ETF
    "IEF",   # iShares 7–10 Year Treasury Bond ETF
    "SHY",   # iShares 1–3 Year Treasury Bond ETF
    "BND",   # Vanguard Total Bond Market ETF
    "AGG",   # iShares Core U.S. Aggregate Bond ETF
    "TIP",   # iShares TIPS Bond ETF (inflation-protected)
    "LQD",   # iShares Investment Grade Corporate Bond ETF
    "HYG",   # iShares High Yield Corporate Bond ETF
    "JNK",   # SPDR Bloomberg High Yield Bond ETF
    "VCIT"   # Vanguard Intermediate-Term Corporate Bond ETF
]

# 🏦 2. U.S. Treasury Yield Indices (not ETFs, yield rates only)
# Note: These are yield indices, not price-traded instruments
treasury_yield_indices = [
    "^IRX",  # U.S. 1Y Treasury yield
    "^FVX",  # U.S. 2Y and 5Y Treasury yield (Yahoo uses the same ticker)
    "^TNX",  # U.S. 10Y Treasury yield
    "^TYX"   # U.S. 30Y Treasury yield
]

# 🌍 3. International Bond ETFs
international_bond_etfs = [
    "BWX",  # International Treasury Bond ETF
    "EMB",  # Emerging Market Bonds
    "IBND"  # International Corporate Bonds
]

# Combined Master List (optional)
tickers = bond_etfs + treasury_yield_indices + international_bond_etfs +tickers1

d_today =dt.date.today()

def yfinance_query(tick, s_date,l_date):
    dictionary = {}
    for i in tickers:
      dictionary[f'{i}'] = yf.download(i, start = s_date, end = l_date )
    return dictionary



def to_pickle(q_data):
    s_dir = script_dir
    with open('full_data.pkl', 'wb') as f:
        pickle.dump(q_data, f)
    
    b_up = s_dir +"//Back Up Data//"
    
    with open( f'{b_up}full_data_back_up_{dt.date.today()}.pkl', 'wb') as b:
        pickle.dump(q_data, b)
    print(f'Data_Save.')

#%%
# This blocks query large amount of data from yfinance and stores it

s_date = dt.date(1980,1,1)
query_data = yfinance_query(tickers, s_date,d_today)
to_pickle(query_data)


#%%

# This only query the latest data
import pickle


if os.path.exists('full_data.pkl'):
    print("File exists.")
    
    with open('full_data.pkl', 'rb') as f:
        f_data = pickle.load(f)

    dates = pd.DataFrame()
    for i in tickers:
        dates[f'{i}']=  f_data[f'{i}']['Close']
    dates = dates.reset_index()
    last_date_query = dates.loc[len(dates)-1,"Date"]
    
else:
    print("File does not exist.")




