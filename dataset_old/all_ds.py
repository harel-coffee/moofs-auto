import os
import pandas as pd
import numpy as np

# Loading data sets
# data_names = {  "hepatitis.csv",
#                 "pima_indian_diabetes.csv",
#                 "heart.csv",
#                 }

print("HEART DISEASE DATA SET")
heart_data = pd.read_csv("datasets/heart.csv")
print(heart_data.columns)
# print(heart_data.head())
heart_data["target"] = heart_data["target"].astype("bool")
print(heart_data.head())
# Data contains categorical variables and .getdummies seperates the different categories of categorical variables as separate binary columns
# a = pd.get_dummies(heart_data['cp'], prefix = "cp")
# b = pd.get_dummies(heart_data['thal'], prefix = "thal")
# c = pd.get_dummies(heart_data['slope'], prefix = "slope")
# frames = [heart_data, a, b, c]
# heart_data = pd.concat(frames, axis = 1)
# heart_data = heart_data.drop(columns = ['cp', 'thal', 'slope'])
# print(heart_data.columns)


print("HEPATITIS DATA SET")
missing = ["na", "--", ".", ".."]
hepatitis_data = pd.read_csv("datasets/hepatitis.csv", na_values=missing)
print(hepatitis_data.columns)
print(hepatitis_data.head())
hepatitis_data["class"].replace((1, 2), (0, 1), inplace=True)
hepatitis_data["class"] = hepatitis_data["class"].astype("bool")
print(hepatitis_data.head())
# Data contains categorical variables and .getdummies seperates the different categories of categorical variables as separate binary columns
# hepatitis_data1 = pd.get_dummies(hepatitis_data, drop_first=True)
# print(hepatitis_data1.columns)

print("PIMA INDIAN DIABETES DATA SET")
diabetes_data = pd.read_csv("datasets/pima_indian_diabetes.csv")
print(diabetes_data.columns)
print(diabetes_data.head())
