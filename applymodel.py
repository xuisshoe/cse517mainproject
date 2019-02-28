#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:18:56 2019

@author: xuisshoe
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#import seaborn as sns
from sklearn import linear_model
#----------- read data ---------------

testF = pd.read_csv("test.csv")
#----------- read final test data ---------------
testF["Climate_Zone"] = np.trunc(testF["Soil_Type"]/1000)
#print(fireSpot["Climate_Zone"])
testF["Geologic_Zone"] = np.trunc(testF["Soil_Type"]/100)%10
#print(fireSpot["Geologic_Zone"])

one_heat_CZ_test = pd.get_dummies(testF["Climate_Zone"] )
one_heat_GZ_test = pd.get_dummies(testF["Geologic_Zone"] )
a_test = np.array(one_heat_CZ_test)
b_test = np.array(one_heat_GZ_test)
testF["one_heat_CZ_4"] = a_test[:,0]
testF["one_heat_CZ_6"] = a_test[:,1]
testF["one_heat_CZ_7"] = a_test[:,2]
testF["one_heat_CZ_8"] = a_test[:,3]
testF["one_heat_GZ_1"] = b_test[:,0]
testF["one_heat_GZ_2"] = b_test[:,1]
testF["one_heat_GZ_7"] = b_test[:,2]
x_final_test = testF[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
                    "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","one_heat_CZ_4",\
                    "one_heat_CZ_6","one_heat_CZ_7","one_heat_CZ_8","one_heat_GZ_1","one_heat_GZ_2","one_heat_GZ_7"]]