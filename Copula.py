# -*- coding: utf-8 -*-
"""
Created on Sun May 25 22:13:21 2025

@author: patrick
"""

import numpy as np
from scipy.stats import norm,t,cauchy
import matplotlib.pyplot as plt
import sys
import os

file_dir = os.getcwd() ;print(file_dir)
file_loc = os.path.dirname(__file__)
changing_file_dir = os.chdir(file_loc)
file_dir = os.getcwd() ;print(file_dir)


def generating_rand(row,col = 0,dist = "rand"):
    rand_type= getattr(np.random, dist)
    rand_val = rand_type(row,col)
    return rand_val


gen_rand = generating_rand(5,10)



