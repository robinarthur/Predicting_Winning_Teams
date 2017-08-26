"""
works on my machine only under linux
the goal is to predict full time result - FTR
"""
# data preprocessing
import pandas as pd
# produces a prediction model in the form of an ensemble of weak prediction
# models, typically decision tree
import xgboost as xgb
# the outcome (dependent variable) has only a limited number of possible values.
# Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
