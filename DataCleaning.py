#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:24:05 2020

@author: praveen
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------------------------------------------------------- #

#load df
df = pd.read_csv("/Users/praveenvudumu/desktop/datascience/datasets/auto-mpg.csv")

#get data types and array size
df.dtypes
df.shape

#replace '?' with NaN
df = df.replace('?', np.nan)

#find NaN sum
df.isna().sum()

#fill NaN with mean()
df['MPG'] = df['MPG'].fillna(df['MPG'].astype(float).mean())
df['Displacement'] = df['Displacement'].fillna(df['Displacement'].astype(float).mean())
df['Weight'] = df['Weight'].fillna(df['Weight'].astype(float).mean())
#df['MPG'] = df['MPG'].fillna(df['MPG'].mean())

#drop all NaN since number of missing values are less now
df = df.dropna()

#drop 'car name' irrelevant column
df.drop(['Model','bore','stroke','compression-ratio'], axis = 1, inplace = True)

#find values numeric in nature
df['Year'].str.isnumeric().value_counts()
df['Year'].loc[df['Year'].str.isnumeric() == False]

#extract first four characters from 'Year'
df_ext = df['Year'].str.extract(r'^(\d{4})', expand=False)

df['Year'].isnull().values.any() #confirm no missing values
df['Year'] = pd.to_numeric(df_ext) #change datatype to numberic and assign to df

#Caluclate 'Age'
df['Age'] = dt.datetime.now().year - df['Year']
df.drop(['Year'], axis = 1, inplace = True) # drop 'Year'

#convert datatypes
df.dtypes

df['MPG'] = pd.to_numeric(df['MPG'], errors='coerce') #coerce-invalid parsing NaN
df['Displacement'] = pd.to_numeric(df['Displacement'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

df['Acceleration'].loc[df['Acceleration'].str.isnumeric() == False] #no no-numeric values
df['Acceleration'] = pd.to_numeric(df['Acceleration'], errors='coerce')
df['Horsepower'].loc[df['Horsepower'].str.isnumeric() == False] #no no-numeric values
df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors='coerce')

df['Cylinders'].loc[df['Cylinders'].str.isnumeric() == False] #find what records are not numeric
df_cylinders = df['Cylinders'].loc[df['Cylinders'] != '-'] #extract all numeric values
cmean = df_cylinders.astype(int).mean() #convert to integer and calc mean
df['Cylinders'] = df['Cylinders'].replace('-', cmean).astype(int) #replace missing values with mean

df['Origin'].unique()
df['Origin'] = np.where(df['Origin'].str.contains('US'), 'US', df['Origin']) #convert and replace if records contains 'US'
df['Origin'] = np.where(df['Origin'].str.contains('Europe'), 'Europe', df['Origin'])
df['Origin'] = np.where(df['Origin'].str.contains('Japan'), 'Japan', df['Origin'])

df.drop(['Cylinders', 'Origin'], axis = 1, inplace = True) #drop these features because they are discrete and not continuos; optional use one-hot encoding?

# EDA

plt.bar(df['Age'], df['MPG']) #bar plot
plt.xlabel('Age')
plt.ylabel('Miles per gallon')

df.plot.scatter(x='Weight',
                y='Acceleration',
                c='Horsepower',
                colormap='viridis',
                ); #scatter plot

df_corr = df.corr() #correlation matrix and heatmap
df_corr
sns.heatmap(df_corr, annot=True, cmap = plt.cm.nipy_spectral_r)

#box plot to detect outliners
df.boxplot(column = ['Age'])

# Standardize the 'Age' feature; mean = 0 and variance = 1
scalar = StandardScaler()
age_scaled = scalar.fit_transform(df["Age"].values.reshape(-1,1))
df["Age_Scaled"] = age_scaled #add to df
df.boxplot(column = ['Age_Scaled'])
age_outliner = df.loc[df["Age_Scaled"] > 3] #outliners can be records whose StandDev is more than 3
age_outliner #view Age records with StdDev > 3
#set outliners to mean
df[((df['Age'] >= 120) & (df['Age'] <= -1))] = df["Age"].mean()
df.boxplot(column = ["Age"])

# Regression - Normalize
X = df.drop('MPG', axis=1)
y = df['MPG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
linear_model = LinearRegression(normalize = True).fit(X_train, y_train)
print('Training score: ', linear_model.score(X_train, y_train))
y_pred = linear_model.predict(X_test)
#print('Testing score: ', r2_score(y_test, y_pred)) # Convert all to numeric (one-hot encoding)

# -------------------------------------------------------------------------------------------------------------- #

# Univarient Feature Imputation (Impute/Inferring missing values from current data - constant, mean, median, mode)

from sklearn.impute import SimpleImputer

df = pd.read_csv("/Users/praveenvudumu/desktop/datascience/datasets/diabetes.csv")
#get basic stats of df
df.shape
df.info()
df.describe().transpose() #Min value of 0 for features in this df indicates missing values; working with data in real world, we should understand how missing values is represented in df

#replace 0 with NaN
df['Glucose'].replace(0, np.nan, inplace= True)
df['BloodPressure'].replace(0, np.nan, inplace= True)
df['SkinThickness'].replace(0, np.nan, inplace= True)
df['Insulin'].replace(0, np.nan, inplace= True)
df['BMI'].replace(0, np.nan, inplace= True)

df.isnull().sum() #find missing values

df_skinthickness = df['SkinThickness'].values.reshape(-1,1) #reshape feature in 2D array
df_skinthickness.shape

imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent') #mode to fill missing values
imp.fit(df['SkinThickness'].values.reshape(-1,1)) #fit on existing values
df['SkinThickness'] = imp.transform(df['SkinThickness'].values.reshape(-1,1)) #transform will fill missing values

df['SkinThickness'].describe() #stats change

imp = SimpleImputer(missing_values = np.nan, strategy='median') #median to fill missing values; 50th percentile of data
imp.fit(df['Glucose'].values.reshape(-1,1))
df['Glucose'] = imp.transform(df['Glucose'].values.reshape(-1,1))

imp = SimpleImputer(missing_values = np.nan, strategy='mean') #mean/avg to fill missing values
imp.fit(df['BloodPressure'].values.reshape(-1,1))
df['BloodPressure'] = imp.transform(df['BloodPressure'].values.reshape(-1,1))

imp = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value = 32) #constant 32 to fill missing values
imp.fit(df['BMI'].values.reshape(-1,1))
df['BMI'] = imp.transform(df['BMI'].values.reshape(-1,1))

# -------------------------------------------------------------------------------------------------------------- #

# Multivariant Feature Imputation (Use entire set of available features to estimate the missing values and not just that missing value feature alone)
# Models each feature with missing calues as a function of other features in an iterative round-robbin fashion and fits a regressor to find missing values

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_features = df.drop('Outcome', axis = 1) #drop Y lable to calc the missing values in Inulin feature using all other features
df_label = df['Outcome']

imp = IterativeImputer(max_iter = 1000, random_state = 42)
imp.fit(df_features)

df_features_arr = imp.transform(df_features) #array with all missing values filled in
df_features = pd.DataFrame(df_features_arr, columns = df_features.columns) #convert array to pd df to have same columns

df = pd.concat([df_features, df_label], axis = 1)#concat the features and labels to one df

# -------------------------------------------------------------------------------------------------------------- #
"""
#pandas to count +/- infinity
pd.options.mode.use_inf_as_na = True

#create a dataset with NaN
df = pd.DataFrame({"SNo" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "Sales" : [np.nan, -np.inf, 450, np.nan, np.nan, 200, np.nan, 110, np.nan, np.nan],
                    "RandomNum":[9, np.nan, np.nan, 1, 7, np.nan, np.inf, np.nan, np.nan, 2 ]
                    })

#df with boolean values indicating whether a value was null or not
df.isnull()

#count of null values in each column
df.isnull().sum()

#replace NaN with 0
df.replace(np.nan, 0)

#fill NaN with last valid observation ## Only used when data is Sorted
df.fillna(method="pad")
df.fillna(method="ffill")
#df.fillna(method="pad", limit= 1)
#fill NaN with next valid observation to fill gap
df.fillna(method="bfill")

#drop NaN Rows
df.dropna(axis = 0 )

#drop NaN Columns
df.dropna(axis = 1 )

#fill NaN values with the mean
df.fillna(df.mean())

"""