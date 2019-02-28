#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:51:23 2019

@author: xuisshoe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import seaborn as sns
from sklearn.linear_model import Ridge

fireSpot = pd.read_csv('train.csv')
testF = pd.read_csv("test.csv")

#fireSpot.head()
#fireSpot.info()
#fireSpot.describe()
#fireSpot.columns
#sns.pairplot(fireSpot)
#sns.distplot(fireSpot['Horizontal_Distance_To_Fire_Points'])
#plt.show()

#plt.hist(fireSpot['Horizontal_Distance_To_Fire_Points'])
#plt.show()

fireSpot["Hillshade_mean"] = (fireSpot["Hillshade_9am"] + fireSpot["Hillshade_Noon"] + fireSpot["Hillshade_3pm"]) / 3
fireSpot["log_elevation"] = np.log(fireSpot["Elevation"])
fireSpot["Euclidean_Distance_To_Hydrology"] = np.sqrt(fireSpot["Horizontal_Distance_To_Hydrology"] ** 2 + fireSpot["Vertical_Distance_To_Hydrology"] ** 2)
fireSpot["Hillshade_9am_sq"] = np.square(fireSpot["Hillshade_9am"])
fireSpot["Hillshade_noon_sq"] = np.square(fireSpot["Hillshade_Noon"])
fireSpot["Hillshade_3pm_sq"] = np.square(fireSpot["Hillshade_3pm"])
fireSpot["cosine_slope"] = np.cos(fireSpot["Slope"])
fireSpot["interaction_9amnoon"] = fireSpot["Hillshade_9am"] * fireSpot["Hillshade_Noon"]
fireSpot["interaction_9amnoon"] = fireSpot["Hillshade_3pm"] * fireSpot["Hillshade_Noon"]
fireSpot["interaction_9amnoon"] = fireSpot["Hillshade_9am"] * fireSpot["Hillshade_3pm"]


X_train = fireSpot[["Hillshade_mean","log_elevation","Euclidean_Distance_To_Hydrology","Hillshade_9am_sq","Hillshade_noon_sq","Hillshade_3pm_sq","cosine_slope","interaction_9amnoon","interaction_9amnoon","interaction_9amnoon"]]
y_train = fireSpot["Horizontal_Distance_To_Fire_Points"]


#X_train = fireSpot[["Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Soil_Type"]]
#y_train = fireSpot["Horizontal_Distance_To_Fire_Points"]

ID = testF["ID"]

testF["Hillshade_mean"] = (testF["Hillshade_9am"] + testF["Hillshade_Noon"] + testF["Hillshade_3pm"]) / 3
testF["log_elevation"] = np.log(testF["Elevation"])
testF["Euclidean_Distance_To_Hydrology"] = np.sqrt(testF["Horizontal_Distance_To_Hydrology"] ** 2 + testF["Vertical_Distance_To_Hydrology"] ** 2)
testF["Hillshade_9am_sq"] = np.square(testF["Hillshade_9am"])
testF["Hillshade_noon_sq"] = np.square(testF["Hillshade_Noon"])
testF["Hillshade_3pm_sq"] = np.square(testF["Hillshade_3pm"])
testF["cosine_slope"] = np.cos(testF["Slope"])
testF["interaction_9amnoon"] = testF["Hillshade_9am"] * testF["Hillshade_Noon"]
testF["interaction_9amnoon"] = testF["Hillshade_3pm"] * testF["Hillshade_Noon"]
testF["interaction_9amnoon"] = testF["Hillshade_9am"] * testF["Hillshade_3pm"]


#X_test = testF[["ID","Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]
#print(len(X_test))

X_test = testF[["Hillshade_mean","log_elevation","Euclidean_Distance_To_Hydrology","Hillshade_9am_sq","Hillshade_noon_sq","Hillshade_3pm_sq","cosine_slope","interaction_9amnoon","interaction_9amnoon","interaction_9amnoon"]]
#y_train = fireSpot["Horizontal_Distance_To_Fire_Points"]

#X_test = testF[["Horizontal_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Soil_Type"]]
#print(len(X_test))
#lm = LinearRegression(normalize = True)
ri = Ridge()
#lg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
#lm.fit(X_train,y_train)
#lg.fit(X_train,y_train)
ri.fit(X_train,y_train)
#predictions = lm.predict(X_test)
#predictions = lg.predict(X_test)
predictions = ri.predict(X_test)
df = pd.DataFrame(predictions,index = ID)
plt.hist(predictions)
plt.show()
#print(len(predictions))
print(df)
#dataoutput = pd.DataFrame({'ID':ID,'Horizontal_Distance_To_Fire_Points':predictions})
#dataoutput.to_csv("prediction.csv", index=False)
#print(predictions)

#print("len:",len(predictions))
#plt.scatter(X_test,predictions)
#plt.show()