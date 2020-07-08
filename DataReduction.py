#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:39:44 2020

@author: praveen
"""
# Feature Correlation

import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.target import FeatureCorrelation
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

def build_model_logistic(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = LogisticRegression(solver='liblinear').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("accuracy_score : ", accuracy_score(y_test, y_pred))

def build_model_linear(x, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("r2_score : ", r2_score(y_test, y_pred))

df = pd.read_csv("/Users/praveenvudumu/desktop/datascience/datasets/diabetes.csv")
df_corr = df.corr() #correlation matrix and heatmap
df_corr
sns.heatmap(df_corr, annot = True, cmap = 'YlGnBu')

X = df[['Insulin', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction']]
y = df['Age']
features = X.columns
visualizer = FeatureCorrelation(labels = features, method = 'pearson') #pearson correlations are meaningful only for metric variables (continuous numeric variables) and also dichotomous variable
visualizer.fit(X, y)
visualizer.poof() #plot
visualizer_df = pd.DataFrame( {
    'Features': visualizer.features_,
    'Scores': visualizer.scores_
    })
visualizer_df

X = df.drop('Outcome', axis = 1) #Calc for all features
y = df['Outcome']
features = X.columns
visualizer = FeatureCorrelation(labels = features, method = 'pearson')
visualizer.fit(X,y)
visualizer.poof() #plot
visualizer_df = pd.DataFrame( {
    'Features': visualizer.features_,
    'Scores': visualizer.scores_
    })
visualizer_df

# -------------------------------------------------------------------------------------------------------------- #

# Mutual Information - Classification https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html
## It represents the dependency between two variables - 0 when independent and 1 when dependent
discrete = [False for _ in range(len(features))] #boolean vector with T/F values to represent if a feature is discrete or not; initialize this as False assuming all features are continuous
discrete[0] = True #set feature[0, Pregnancies] = T
visualizer = FeatureCorrelation(method = 'mutual_info-classification',
                                labels = features)
visualizer.fit(X, y, discrete_features = discrete)
visualizer.poof()

features = ['Age', 'BMI', 'Insulin', 'BloodPressure', 'Glucose'] #create list of features to plot
visualizer = FeatureCorrelation(method = 'mutual_info-classification',
                                feature_names = features, sort = True)
visualizer.fit(X, y)
visualizer.poof()

# -------------------------------------------------------------------------------------------------------------- #

# Multicollinearity
#Performing regression analysis using data that is multicollinear, that is one or more of the input features can be represented as linear combinations of other features; these models are not very robust

df_auto #cleaned df from DataCleaning.py
df_auto.describe() #all on different scales - should standardize

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df_auto[['Displacement']] = preprocessing.scale(df_auto[['Displacement']].astype('float64')) #standardize because all features are on different scales this will help records center around zero
df_auto[['Horsepower']] = preprocessing.scale(df_auto[['Horsepower']].astype('float64'))
df_auto[['Weight']] = preprocessing.scale(df_auto[['Weight']].astype('float64'))
df_auto[['Acceleration']] = preprocessing.scale(df_auto[['Acceleration']].astype('float64'))
df_auto[['Age']] = preprocessing.scale(df_auto[['Age']].astype('float64'))

X = df_auto.drop(['MPG', 'Origin'], axis = 1) #Origin because Categorical, can use one-hot encoding to convert
y = df_auto['MPG']

build_model_linear(X, y)

#When there are multiple features in regression model, adjusted r2 score is a better measure (corrected goodness of fit)
def adjusted_r2(r2, labels, features):
    adj_r2 = 1 - ((1 - r2) * (len(labels) - 1)) / (len(labels) - features.shape[1] - 1)
    return adj_r2
adjusted_r2(r2_score(y_test,y_pred), y_test, X_test)
features_corr = X.corr() #cyclider, horsepower, weight are highly correlated with displacement of a car, corr() coeff > 0.9; this high correlation coefficient almost at .9 and aboce indicates that these variables are likely to be colinear
# Another way to think about this is that all of these variables - cylinders, horsepower, weight, and displacement give us the same information so we don't really need to use all of them in our regression analysis

abs(features_corr) > 0.8

# to avoid multicollinearity that exists in these feature variables, we can drop the features - cylinders, displacement, weight and leave only horsepower
trim_features_df = X.drop(['Cylinders', 'Displacement', 'Weight'], axis = 1)
trim_features_df.corr() #corr coeff low

# -------------------------------------------------------------------------------------------------------------- #

# VIF - Variance Inflation Factor: a measure to quantify the severity of multicollinearity in an OLS (ordinary least square) regression

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i)
                     for i in range(X.shape[1])] #VIF factor for a particular feature i is calculated as a relationship between all other features, that is all features other than i, and that feature i; we calculate VIF for each feature separately. so for a particular feature i, use all of the other values to calculate this VIF Factor.
vif["features"] = X.columns
vif.round(3) # VIF 1:not correlated, 1-5:moderately correlated, >5:highly correlated

X = X.drop(['Displacement', 'Weight'], axis = 1) #VIF>5, high correlation
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i)
                     for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(3)

X = df_auto.drop(['MPG', 'Displacement','Weight', 'Origin'], axis = 1) #build LR model with features VIF<5
y = df_auto['MPG']

build_model_linear(X, y)
adjusted_r2(r2_score(y_test,y_pred), y_test, X_test) #when there are no highly correlated features, r2 is very close to adj_r2

# -------------------------------------------------------------------------------------------------------------- #

# Feature Selection - Missing Values

df_diabetes #cleaned df from DataCleaning.py
df_diabetes_trimmed = df_diabetes.dropna(thresh = int(df_diabetes.shape[0] * 0.9), axis = 1) #only retain columns that have more than 90% records present
df_diabetes_trimmed.columns


# Feature Selection - Variance Threshold

X = df_diabetes.drop('Outcome', axis = 1)
y = df_diabetes['Outcome']
X.var(axis = 0) #variance of all X features; less value may infer that feature has more infomrmation

from sklearn.preprocessing import minmax_scale # scaling becuase features are on different scales
X_scaled = pd.DataFrame(minmax_scale(X, feature_range = (0,10)), columns = X.columns)
X_scaled.var(axis = 0)

from sklearn.feature_selection import VarianceThreshold #feature selector that removes all low-variance features
varthreshold = VarianceThreshold(threshold = 2) #features with a high variance have more predictive power and contain more information, then we choose variance threshold estimator to select features; only choose var() > 1
X_new = varthreshold.fit_transform(X_scaled)
varthreshold.get_support()
X_scaled.columns[varthreshold.get_support()] #display features which have var() more than threshold

df_highVar = df_diabetes[['Pregnancies', 'Glucose', 'Age', 'Outcome']] #create a new df with high var features to build ML model
X_highVar = df_highVar.drop(['Outcome'], axis = 1)
y_highVar = df_highVar['Outcome']

build_model_logistic(X, y)
build_model_logistic(X_highVar, y_highVar) #better score


# Feature Selection - Statistical Techniques

# Chi-Sqr: chi-square test is used in statistics to test the independence of two events. Given the data of two  variables, we can get observed count O and expected count E. Chi-Square measures how expected count E and observed count O deviate from each other. typically used in regression models

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import f_classif, SelectPercentile

df_heart = pd.read_csv('/Users/praveenvudumu/desktop/datascience/datasets/heatdisease.csv')
X = df_heart.drop(["target"], axis = 1)
y = df_heart["target"]
test = SelectKBest(score_func = chi2, k = 5)
fit = test.fit(X, y) #using all features to find relevant features and not perfoming any train/test
feature_score = pd.DataFrame() #get features and scores in a seperate df
for i in range(X.shape[1]):
    new = pd.DataFrame({'Features' : X.columns[i],
                        'ChiSqr Score' : fit.scores_[i]}, index=[i])
    feature_score = pd.concat([feature_score, new])
feature_score = feature_score.sort_values(by="ChiSqr Score")
feature_score #figher scores more relevant features for ML model
X_new = fit.transform(X)
fit.get_support()
X.columns[fit.get_support()]

X_chi2_features = pd.DataFrame(X_new, columns = X.columns[fit.get_support()]) #k=5, store 5 best features with max chi2 scores


# ANOVA
# Select features according to a percentile of the highest scores; typically used in classification models

test = SelectPercentile(f_classif, percentile = 25) #features in top 25th percentile
fit = test.fit(X, y)
feature_score = pd.DataFrame() #get features and scores in a seperate df
for i in range(X.shape[1]):
    new = pd.DataFrame({'Features' : X.columns[i],
                        'Score' : fit.scores_[i]}, index=[i])
    feature_score = pd.concat([feature_score, new])
feature_score = feature_score.sort_values(by="Score")
feature_score #figher scores more relevant features for ML model
X_new = fit.transform(X)
fit.get_support()
X.columns[fit.get_support()]

X_anova_features = pd.DataFrame(X_new, columns = X.columns[fit.get_support()]) #only 3 features in top 25th percentile

build_model_logistic(X, y) #all features
build_model_logistic(X_chi2_features, y) #5 features
build_model_logistic(X_anova_features, y) #3 features

# -------------------------------------------------------------------------------------------------------------- #

# Wrapper Methods
# Recursive Feature Elimination(RFE)-select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute; then, the least important features are pruned from current set of features and this procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector


model = RandomForestClassifier(n_estimators = 10) #10 models within
rfe = RFE(model, n_features_to_select = 5)
fit = rfe.fit(X, y)

feature_rank = pd.DataFrame({
                            'features': X.columns,
                            'selected': fit.support_,
                            'feature rank': fit.ranking_})
feature_rank = feature_rank.sort_values(by = "feature rank")
feature_rank #top 5 features with rank = 1 and these are the most relevant features
selected_feature_names = feature_rank.loc[feature_rank["selected"] == True]
selected_feature_names #seleted top 5 features
X_selected_features = X[selected_feature_names["features"].values]

# -------------------------------------------------------------------------------------------------------------- #

# Sequential Feature Selector (SFA) - remove or add one feature at the time based on the classifier performance until a feature subset of the desired size k is reached
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#:~:text=Overview,feature%20subspace%20where%20k%20%3C%20d

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators = 10),
                                             k_features = 5,
                                             forward = True, #add features till model imporves
                                             scoring = "accuracy",
                                             cv = 4)
features = feature_selector.fit(np.array(X), y)
forward_elimination_feature_names = list(X.columns[list(features.k_feature_idx_)])
forward_elimination_feature_names #5 most important features
X_forward_elimination_features = X[forward_elimination_feature_names]

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators = 10),
                                             k_features = 5,
                                             forward = False, #drop features till model improves
                                             scoring = "accuracy",
                                             cv = 4)
features = feature_selector.fit(np.array(X), y)
backward_elimination_feature_names = list(X.columns[list(features.k_feature_idx_)])
backward_elimination_feature_names
X_backward_elimination_features = X[forward_elimination_feature_names]

build_model(X,y) #baseline
build_model(X_selected_features, y) #RFE
build_model(X_forward_elimination_features, y) #SFE_forward
build_model(X_backward_elimination_features, y) #SFE_backward

# -------------------------------------------------------------------------------------------------------------- #

# Embedded Methods
# ML model by itself assigns feature importance

df_carprice = pd.read_csv("/Users/praveenvudumu/desktop/datascience/datasets/car_price_data.csv")
df_carprice.drop(
    ["car_ID", "symboling", "CarName", "aspiration", "carbody", "enginelocation", "enginetype",     "fuelsystem", "cylindernumber"], axis = 1, inplace = True
    ) #drop irrelevant features
df_carprice["drivewheel"].unique() #get unique features to label encode or one-hot encode
df_carprice["doornumber"].unique()
df_carprice["fueltype"].unique()

doornumber_dict = {'two':0, 'four':1} #label encoding
df_carprice['doornumber'].replace(doornumber_dict, inplace=True)
df_carprice = pd.get_dummies(df_carprice, columns=['drivewheel']) #one-hot encoding
df_carprice = pd.get_dummies(df_carprice, columns=['fueltype'])

X = df_carprice.drop("price", axis = 1)
y = df_carprice["price"]

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.8) #hyperparameter 0.8; penalty for complex coeff
lasso.fit(X, y)
predictors = X.columns
coef = pd.Series(lasso.coef_, predictors).sort_values()
print(coef) #coeff with large absolute values more important
lasso_features = ["stroke", "boreratio", "drivewheel_fwd", "drivewheel_rwd", "carwidth"]
X[lasso_features].head()

from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(max_depth = 5)
decision_tree.fit(X, y)
predictors = X.columns
coef = pd.Series(decision_tree.feature_importances_, predictors).sort_values()
print(coef) #coeff with highest weights are important
decision_tree_features = ["enginesize", "curbweight", "highwaympg"]
X[decision_tree_features].head()

build_model_linear(X[lasso_features], y)
build_model_linear(X[decision_tree_features], y)

# -------------------------------------------------------------------------------------------------------------- #

# Dimensionality Reduction

# PCA

import statsmodels.api as sm
from sklearn.decomposition import PCA

df_carprice #pre-process this df above
X = df_carprice.drop("price", axis = 1)
y = df_carprice["price"]

build_model_linear(X, y)
features = list(X.columns) #total features 19
# features
# len(features)
def apply_pca(X, n): #n- no.of PCA
    pca = PCA(n_components = n)
    X_new = pca.fit_transform(X)
    return pca, pd.DataFrame(X_new) #function to apply PCA to input features

pca, _ = apply_pca(df_carprice, len(features)) #all features
print ("explained variance:", pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_)) #most of the variance is captured in first two PCA
plt.xlabel("n components")
plt.ylabel("cumulative variance")

pca = PCA(n_components = 4) #change based on pot above
X_new = pca.fit_transform(X, y)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
X_new = pd.DataFrame(X_new)
#X_new.head()
build_model_linear(X_new, y) # compare with base model and with different n_components

# -------------------------------------------------------------------------------------------------------------- #

# LDA: new axis of projection to maximize the seperation between the data points
#dimensionality for classification models

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df_diabetes #outcome is categorical
X = df_diabetes.drop('Outcome', axis = 1) #Calc for all features
y = df_diabetes['Outcome']

lda = LinearDiscriminantAnalysis(n_components = 2) #first two dimensions
X_new = lda.fit_transform(X, y)
lda.explained_variance_ratio_
sum(lda.explained_variance_ratio_) #100% of this variance is explained by first two LDA components
X_new = pd.DataFrame(X_new)
X_new.head()

# -------------------------------------------------------------------------------------------------------------- #

# Manifold Learning
# Comparison of Manifold Learning methods - https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
# Kernel PCA
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

# -------------------------------------------------------------------------------------------------------------- #




