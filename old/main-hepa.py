# Import basic libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.preprocessing as skp
import sklearn.model_selection as skm

# import classification modules
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Selection
from sklearn.model_selection import GridSearchCV as gs
from sklearn.model_selection import RandomizedSearchCV as rs
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_auc_score,roc_curve, auc, f1_score
import sklearn.metrics as metrics


# Loading Dataset
missing = ["na", "--", ".", ".."]
hepatitis_data = pd.read_csv("datasets/hepatitis.csv", na_values=missing)
# print(hepatitis_data.head())
# print(hepatitis_data.isnull().sum()) # Checking for nulls
hepatitis_data["class"].replace((1, 2), (0, 1), inplace=True)
hepatitis_data["class"] = hepatitis_data["class"].astype("bool")
# print(hepatitis_data.describe())

# Discretization of age Column
# hepatitis_data["age"]=np.where((hepatitis_data["age"]>10) & (hepatitis_data["age"]<20),"Teenagers",
#                    np.where((hepatitis_data["age"]>=20) & (hepatitis_data["age"]<=30),"Adults",
#                    np.where((hepatitis_data["age"]>30) & (hepatitis_data["age"]<=40),"Middle aged",np.where((hepatitis_data["age"]<=10),"Children",
#                             "Old"))))
# hepatitis_data["age"]=pd.Categorical(hepatitis_data.age,["Children",'Teenagers','Adults', 'Middle aged', 'Old'],ordered=True)
# hepatitis_data["age"].value_counts()
# sns.barplot(x="age", y="class", data=hepatitis_data)
# plt.show()

hepatitis_data["sex"].replace((1, 2), ("male", "female"), inplace=True)
hepatitis_data["sex"] = pd.Categorical(hepatitis_data.sex, ["male", 'female'], ordered=False)

# Dropping all nulls
hepatitis_data.dropna(inplace=True)
# print(hepatitis_data.dtypes)

# We have categorical variables .getdummies seperates the different categories of categorical variables as separate binary columns
hepatitis_data1 = pd.get_dummies(hepatitis_data, drop_first=True)
# List of new columns
# print(hepatitis_data1.columns)
# print(hepatitis_data1.head(5))

hepatitis_data1["bilirubin"] = np.abs((hepatitis_data1["bilirubin"]-hepatitis_data1["bilirubin"].mean())/(hepatitis_data1["bilirubin"].std()))
hepatitis_data1["albumin"] = np.abs((hepatitis_data1["albumin"]-hepatitis_data1["albumin"].mean())/(hepatitis_data1["albumin"].std()))

y = hepatitis_data1["class"].copy()
X = hepatitis_data1.drop(columns=["class"])
# print(y.shape)
# print(X.shape)

# Random Forest method for feature selection
# =============
model = RandomForestClassifier()
# Get the feature importance with simple steps:
X_features = X.columns
model.fit(X, y)
# display the relative importance of each attribute
importances = np.around(model.feature_importances_, decimals=4)
imp_features = model.feature_importances_
feature_array = np.array(X_features)
sorted_features = pd.DataFrame(list(zip(feature_array, imp_features))).sort_values(by=1, ascending=False)
data_top = sorted_features[:X.shape[1]]
feature_to_rem = sorted_features[X.shape[1]:]
# print("Unimportant Columms after simple Random Forrest\n",feature_to_rem)
rem_index = list(feature_to_rem.index)
# print(rem_index)
# print("Important Columms after simple Random Forrest\n",data_top)
data_top_index = list(data_top.index)
# print("Important Columms after simple Random Forrest\n",data_top_index )
# print(importances)
X_randfor_sel = X.drop(X.columns[rem_index], axis=1)
features_randfor_select = X_randfor_sel.columns
# print(features_randfor_select)

# Create train-test split parts for manual split
# ===============
trainX, testX, trainy, testy = skm.train_test_split(X, y, test_size=0.25, random_state=99)  # explain random state
print("\n shape of train split: ")
print(trainX.shape, trainy.shape)
print("\n shape of train split: ")
print(testX.shape, testy.shape)
# Making X Scalar for ML algorithms
X = skp.StandardScaler().fit(X).transform(X)

# Support Vector Machine Algorithm
# ========
svm = clf = SVC(gamma="auto", kernel='poly', degree=3)
svm.fit(trainX, trainy)
predictions = svm.predict(testX)
accsvm = accuracy_score(testy, predictions)*100
# print("Accuracy of Support Vector Machine (%): \n", accsvm)
fprsvm, tprsvm, _ = roc_curve(testy, predictions)
aucsvm = auc(fprsvm, tprsvm)*100
# print("AUC OF Support Vector Machine (%): \n", aucsvm)
recallsvm = recall_score(testy, predictions)*100
# print("Recall of Support Vector Machine is: \n",recallsvm)
precsvm = precision_score(testy, predictions)*100
# print("Precision of Support Vector Machine is: \n",precsvm)

# Decision Tree Classifier
# ==============
dt = DecisionTreeClassifier(max_depth=10, criterion="gini")
dt.fit(trainX, trainy)
predictions = dt.predict(testX)
accdt = accuracy_score(testy, predictions)*100
# print("Accuracy of Decision Tree (%): \n",accdt)
fprdt, tprdt, _ = roc_curve(testy, predictions)
aucdt = auc(fprdt, tprdt)*100
# print("AUC OF Decision Tree (%): \n",aucdt)
recalldt = recall_score(testy, predictions)*100
# print("Recall of Decision Tree is: \n",recalldt)
precdt = precision_score(testy, predictions)*100
# print("Precision of Decision Tree is: \n",precdt)

# Comparison
# ===========
algos = ["Support Vector Machine", "Decision Tree"]
acc = [accsvm, accdt]
auc = [aucsvm, aucdt]
recall = [recallsvm, recalldt]
prec = [precsvm, precdt]
comp = {"Algorithms": algos, "Accuracies": acc, "Area Under the Curve": auc, "Recall": recall, "Precision": prec}
compdf = pd.DataFrame(comp)
print(compdf)
# ROC
# =======
roc_auc1 = metrics.auc(fprsvm, tprsvm)
roc_auc2 = metrics.auc(fprdt, tprdt)
plt.figure(figsize=(20, 10))
plt.plot(fprsvm, tprsvm, "k", label="ROC of SVM = %0.2f" % roc_auc1)
plt.plot(fprdt, tprdt, "m", label="ROC of Descision Tree= %0.2f" % roc_auc2)
plt.rcParams.update({'font.size': 16})
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=22)
plt.show()

# Tune hyper parameters of DTC
# Decision Tree with random search
parameters = {"min_samples_split": range(10, 200, 10), "max_depth": range(1, 20, 1)}
clf_treers = DecisionTreeClassifier()
clfrs = rs(clf_treers, parameters, cv=5, scoring="precision")
clfrs.fit(trainX, trainy)
predictions = clfrs.predict(testX)
accdtrs = accuracy_score(testy, predictions)*100
print("Accuracy of Decision Tree after Hyperparameter Tuning (%): \n", accdtrs)
fprdtrs, tprdtrs, _ = roc_curve(testy, predictions)
recalldtrs = recall_score(testy, predictions)*100
print("Recall of Decision Tree after Hyperparameter Tuning is: \n", recalldtrs)
precdtrs = precision_score(testy, predictions)*100
print("Precision of Decision Tree after Hyperparameter Tuning is: \n", precdtrs)

# examnine the best model
# single best score achieved accross all params
print("Best Score (%): \n", clfrs.best_score_*100)
# Dictionary Containing the parameters
print("Best Parameters: \n", clfrs.best_params_)
print("Best Estimators: \n", clfrs.best_estimator_)
