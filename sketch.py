#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:53:34 2019

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
fireSpot = pd.read_csv('train.csv')
testF = pd.read_csv("test.csv")
ID = testF["ID"]
#----------- partition data ---------------
#X_train = fireSpot[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",\
#                    "Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]

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
y = fireSpot["Horizontal_Distance_To_Fire_Points"] # select label
#ETC = ExtraTreesClassifier()
#ETC = ETC.fit(X, y)
#model = SelectFromModel(ETC, prefit=True)
#X = model.transform(X) # new feature

#---------- split data -------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

#---------- apply kernel ridge -------------
from sklearn.kernel_ridge import KernelRidge
clf = KernelRidge()  # degree=10 --> score=0.61
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
#plt.scatter(predict, y_test)
score = clf.score(X_test,y_test)
model = clf.get_params()

##------- apply svm -------------
#from sklearn.svm import SVR
#clf = SVR(gamma='scale',C=1.0, epsilon=0.2)
#clf.fit(X_train,y_train)
#predict = clf.predict(X_test)
##plt.scatter(predict, y_test)
#score = clf.score(X_test,y_test)

# ----------- modify test data ----------
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

final_prediction = clf.predict(x_final_test)
dataoutput = pd.DataFrame({'ID':ID,'Horizontal_Distance_To_Fire_Points':final_prediction})
dataoutput.to_csv("prediction.csv", index=False)



