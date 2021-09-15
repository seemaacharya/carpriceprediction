# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:12:24 2021

@author: DELL
"""

#Business requirement-To predict the price of the car
#importing the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
df=pd.read_csv("car data.csv")

#EDA
df.shape
df.info

#To check categorical features
print(df["Fuel_Type"].unique())
#['Petrol' 'Diesel' 'CNG']
print(df["Seller_Type"].unique())
#['Dealer' 'Individual']
print(df["Transmission"].unique())
#['Manual' 'Automatic']
print(df["Owner"].unique())
#[0 1 3]

#Checking the missing values
df.isna().sum()
#no missing values
df.describe()
df.columns

final_dataset=df[["Year", 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()

final_dataset["Current Year"]=2021
final_dataset.head()

final_dataset["no_year"]=final_dataset["Current Year"]-final_dataset["Year"]
final_dataset.head()
final_dataset.drop(["Year"],axis=1,inplace=True)
final_dataset.head()

final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset.columns
final_dataset=final_dataset.drop(["Current Year"],axis=1)
final_dataset.head()

#correlation
final_dataset.corr()
import seaborn as sns
sns.pairplot(final_dataset)

#to get the correlation of each feature in the dataset
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
#plot heatmap
g=sns.heatmap(df[top_corr_features].corr(),annot=True)

#X and Y
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

#Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

#plot of feature importances for better visualization
import matplotlib.pyplot as plt
feat_importances=pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

#train_test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

#Model building
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

#RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)

sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)

from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')
# dump information to that file
pickle.dump(rf_random, file)










