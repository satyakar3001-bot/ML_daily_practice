# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:18:51 2020

@author: asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = train.append(test)
X = data.iloc[: , 0:10].values
y = data.iloc[:, 10].values


M = 5