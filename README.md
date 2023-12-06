# Singapore-resale-flat-price
# Introduction :
         This project aims to construct a machine learning model and implement it as a user-friendly online 
application in order to provide accurate predictions about the resale values of apartments in Singapore.This prediction model will be based on past transactions involving resale flats, 
and its goal is to the future buyers and sellers in evaluating the worth of a flat after it has been previously resale.Resale prices are influenced by a wide variety of criteria, 
including location, the kind of apartment and the total square footage.
# Domain : 
     Real Estate
# Requirement :
      1) Python 
      2) Pandas
      3) Numpy
# Libraries :
      1) import pandas as pd
      2) import numpy as np
      3) import statistics
      4) import seaborn as sns
      5) import matplotlib.pyplot as plt
      6) from sklearn.preprocessing import StandardScaler
      7) from sklearn.model_selection import train_test_split
      8) from sklearn.linear_model import LinearRegression
      9) from sklearn.tree import DecisionTreeRegressor
      10) from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
      11) from sklearn.model_selection import GridSearchCV
      12) import pickle
# Project Workflow :
The following is a fundamental outline of the project:
  1) The Resale Flat Prices dataset has five distinct CSV files, each representing a specific time period. 
These time periods are 1990 to 1999, 2000 to 2012, 2012 to 2014, 2015 to 2016, and 2017 onwards.
  2) The data will be converted into a format that is appropriate for analysis, and any required cleaning and pre-processing procedures will be carried out.
Relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date will be extracted.
Any additional features that may enhance prediction accuracy will also be created.
  3) The objective of this study is to construct a machine learning regression model that utilizes the decision tree regressor to accurately forecast the continuous variable 'resale_price'.

       
