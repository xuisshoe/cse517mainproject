from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

trainf = pd.read_csv('train.csv')
X_train = trainf[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]
y_train = trainf["Horizontal_Distance_To_Fire_Points"]

testf = pd.read_csv('test.csv')
X_ID = testf["ID"]
X_test = testf[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Soil_Type"]]

clf = RandomForestClassifier(max_features = 10, n_estimators = 100)
clf.fit(X_train, y_train)
res = clf.predict(X_test)

for i in range(len(res)):
	print (X_ID[i], res[i])

dataoutput = pd.DataFrame({'ID':X_ID, 'Horizontal_Distance_To_Fire_Points':res})
dataoutput.to_csv("prediction.csv", index=False)
