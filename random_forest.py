#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:55:27 2019

@author: xuisshoe
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

trainf = pd.read_csv('train.csv')
X = trainf[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",\
            "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",\
            "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]
y = trainf["Horizontal_Distance_To_Fire_Points"]
table = X.describe()
X_numerical = trainf[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",\
            "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",\
            "Hillshade_9am","Hillshade_Noon","Hillshade_3pm"]]


model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(X_numerical,y)
print(model.oob_score_)



# ============================= final prediction data ============================
#testf = pd.read_csv('test.csv')
#X_ID = testf["ID"]
#X_test = testf[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]
#
#clf = RandomForestClassifier(max_features = 10, n_estimators = 100)
#clf.fit(X_train, y_train)
#res = clf.predict(X_test)
#
#for i in range(len(res)):
#	print (X_ID[i], res[i])
#
#dataoutput = pd.DataFrame({'ID':X_ID, 'Horizontal_Distance_To_Fire_Points':res})
#dataoutput.to_csv("prediction.csv", index=False)
