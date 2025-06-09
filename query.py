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

# Changing the data location
url = "C://Users//patri//OneDrive//Fin_Data"


#"^CDNX": This ticker does not exist no more

# Tickers that's being query
tickers1 = ["HXQ.TO","VDY.TO","XHU.TO","ZDV.TO","ZDY.TO","ZGQ.TO","ZSP.TO",\
             "^GSPC", "^DJI", "^NDX","^IXIC","^RUT","^VIX",\
                "^GSPTSE", \
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


new_tickers = ["CGL-C.TO", "ZTL.NE", "ZAG.TO", "ZPR.TO", "ZMT.TO",  "XEM.TO", "PSU-U.TO" ]

# Combined Tickers
tickers = bond_etfs + treasury_yield_indices + international_bond_etfs +tickers1 +new_tickers


def yfinance_query(tick, s_date,l_date):
    dictionary = {}
    for i in tickers:
      dictionary[f'{i}'] = yf.download(i, start = s_date, end = l_date )
    return dictionary


def to_pickle(q_data):
    url_l = url
    with open(f'{url_l}//full_data.pkl', 'wb') as f:
        pickle.dump(q_data, f)
    
    b_up = url_l +"//Back Up Data//"
    
    with open( f'{b_up}full_data_back_up_{dt.date.today()}.pkl', 'wb') as b:
        pickle.dump(q_data, b)
    # print(f'Data_Save.')

# This only query the latest data
def query_data(file_name, to_do):
    
    d_today =dt.date.today()
    
    if to_do == "update":
        if os.path.exists(f'{file_name}.pkl'):
            print("File exists.")
            
            with open(f'{file_name}.pkl', 'rb') as f:
                f_data = pickle.load(f)
            
            
            # This delete Keys that are empty
            # keys_to_remove = ["ZTL-NE"]
            # for key in keys_to_remove:
                # del f_data["ZTL-NE"]
                        
            
            for i in tickers:
                
                if i in f_data:
                    print(f'{i}: pass')
                    date = f_data[f'{i}'].reset_index()
                    last_date= date["Date"]
                    last_date = last_date.iloc[-1]
                    new_data= yf.download(i, start = last_date, end = d_today )
                    new_data = new_data.reset_index()
                    new_data = new_data.iloc[1:].set_index("Date")
                    f_data[f'{i}'] = pd.concat([f_data[f'{i}'], new_data], ignore_index=False)
                    data = to_pickle(f_data) # This saves it to a pickle
                else:
                    print(f'{i}: fail')
                    new_data= yf.download(i, start = dt.date(1980,1,1), end = d_today )
                    f_data[f'{i}'] = new_data
                    data = to_pickle(f_data) # This saves it to a pickle
                    print(f'{i} has been added')
                    
            return f_data 
        else:
            return print(f'file does not exist.')
    else:
        s_date = dt.date(1980,1,1)
        query_data = yfinance_query(tickers, s_date,d_today)
        data = to_pickle(query_data)
        with open(f'{file_name}.pkl', 'rb') as f:
            f_data = pickle.load(f)
        return f_data



query = query_data(f'{url}//full_data', to_do='update')





#%%



file_name = "full_data.pkl"

with open(f'{url}//{file_name}', "rb") as f:
    checking = pickle.load(f)

