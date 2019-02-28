#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 22:00:27 2019

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
#X_train = fireSpot[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
#                    "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]
fireSpot["Soil_Type"] = np.trunc(fireSpot["Soil_Type"]/100)
X_train = fireSpot[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
                    "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Soil_Type"]]
y_train = fireSpot["Horizontal_Distance_To_Fire_Points"]
ID = fireSpot["ID"]
#----------- test data ---------------
#X_test = testF[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
#                "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]

#----------- apply model ---------------
#reg = linear_model.Ridge(alpha = 0.5)
reg = linear_model.Lasso(alpha=0.1, max_iter = 10000)
reg.fit(X_train,y_train)
predictions = reg.predict(X_train)
df = pd.DataFrame(predictions,index = ID)
#plt.hist(predictions)
#plt.show()
y_target = np.array([y_train])
y_w = np.array([predictions])
e_in = np.sum(y_target - y_w)

print(df)
print(e_in)
