import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
#import seaborn as sns

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
