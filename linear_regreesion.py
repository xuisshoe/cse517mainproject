#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 18:37:10 2019

@author: xuisshoe
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#import seaborn as sns
from sklearn import linear_model
#----------- read data ---------------
fireSpot = pd.read_csv('train.csv')
testF = pd.read_csv("test.csv")
#----------- partition data ---------------
#X = fireSpot[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
#                   "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]
y = fireSpot["Horizontal_Distance_To_Fire_Points"]

#----------- heat map transformation ---------------
fireSpot["Climate_Zone"] = np.trunc(fireSpot["Soil_Type"]/1000)
#print(fireSpot["Climate_Zone"])
fireSpot["Geologic_Zone"] = np.trunc(fireSpot["Soil_Type"]/100)%10
#print(fireSpot["Geologic_Zone"])

one_heat_CZ = pd.get_dummies(fireSpot["Climate_Zone"] )
one_heat_GZ = pd.get_dummies(fireSpot["Geologic_Zone"] )
a = np.array(one_heat_CZ)
b = np.array(one_heat_GZ)
fireSpot["one_heat_CZ_4"] = a[:,0]
fireSpot["one_heat_CZ_6"] = a[:,1]
fireSpot["one_heat_CZ_7"] = a[:,2]
fireSpot["one_heat_CZ_8"] = a[:,3]
fireSpot["one_heat_GZ_1"] = b[:,0]
fireSpot["one_heat_GZ_2"] = b[:,1]
fireSpot["one_heat_GZ_7"] = b[:,2]
#total_1 = np.sum(one_heat_GZ, axis = 0)
#total_2 = np.sum(one_heat_CZ, axis = 0)
# select feature
X = fireSpot[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
                    "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","one_heat_CZ_4",\
                    "one_heat_CZ_6","one_heat_CZ_7","one_heat_CZ_8","one_heat_GZ_1","one_heat_GZ_2","one_heat_GZ_7"]]
#X = fireSpot[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
#                    "Horizontal_Distance_To_Roadways","Hillshade_Noon","Hillshade_3pm",\
#                    "one_heat_CZ_7","one_heat_CZ_8","one_heat_GZ_2","one_heat_GZ_7"]]

# ------ feature selection ------------
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel
#
#ETC = ExtraTreesClassifier()
#ETC = ETC.fit(X, y)
#model = SelectFromModel(ETC, prefit=True)
#X = model.transform(X) # new feature

#----------- apply model ---------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

from sklearn.linear_model import Ridge
ri = Ridge()
ri.fit(X_train,y_train)
predictions = ri.predict(X_test)
score = ri.score(X_test,y_test)
