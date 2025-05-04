# -*- coding: utf-8 -*-
"""
Created on Sat May  3 17:18:00 2025

@author: patri
"""

import pandas as pd
import numpy as np
import datetime as dt
import pickle
import sys
import os

file_loc = os.path.abspath(__file__)
dir_path = os.path.dirname(file_loc)
os.chdir(dir_path)
print(os.getcwd())


with open("full_data.pkl", 'rb') as r:
    loaded_data = pickle.load(r)


bond = loaded_data["^TYX"]["Close"]
bond = bond.loc[dt.date(2010,1,1):]
