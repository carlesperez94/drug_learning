import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform

# Datia curation
data = pd.read_csv("big_set_2.tsv", sep="\t", decimal=",")
data = data.fillna(0)
droping = ['Molecule ChEMBL ID', 'Set', 'Molecular Weight', 
            '#RO5 Violations', 
            'Score2HB/HA', 
            'Score 2HB filter by size',
             'Simulation sampling',
            'NotDockingResult', 
            'NotSimulationResult',
            'Freq 2HB', 
             'pChEMBL Value',
             'ligprep pose', 
             'Cluster size 2HB', 
             'HB initial pose',
             'Score 1HB',
             'Inactive', 'InactivesUnotsampling',
             'Min BE 2HB', 
             'Sampling after filter 2HB', 
             'Sampling 25% lowest BE', 
             '#heavy atoms', '#rotatable bonds',
            ]

#Columns keept: 'docking score','Score 2HB','NotEnoughSampling','MET438NH mean persistency'

for i in data.iterrows():
    if i[1]["pChEMBL Value"] < 6.5:
        data.at[i[0], "high"] = 0
for i in data.iterrows():
    if i[1]["pChEMBL Value"] >= 6.5:
        data.at[i[0], "high"] = 1
for i in droping:
    data=data.drop(i, axis=1)
print("Selected features:")
print(data.columns[:-1])
X = data.iloc[:,:-1].values
y = data['high']

classifiers = [
               KNeighborsClassifier(),
               SVC(),
               NuSVC(probability=True),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
               AdaBoostClassifier(),
               GradientBoostingClassifier(),
               BaggingClassifier(),
               GaussianNB(),
               MLPClassifier() 
              ]        

pipeline = Pipeline([
    ('normalizer', MinMaxScaler()), #Step1 - normalize data
    ('clf', SVC()) #step2 - classifier
])


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                   y, test_size=0.25, random_state=20)

for classifier in classifiers:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train, cv=10)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
            print(key,' mean ', values.mean())
            print(key,' std ', values.std())

# We have found that SVC with MinMaxScaler is the best one
pipeline.set_params(clf = SVC())
pipeline.steps
params = { 
    'clf__C': [1, 10, 15, 25, 50, 100, 250, 500, 1000],
    'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'clf__gamma' : ['scale','auto'],
    'clf__decision_function_shape' : ['ovo','ovr']
          }
# Finding best params
cv_grid = GridSearchCV(pipeline, param_grid = params)
cv_grid.fit(X_train, y_train)
print(cv_grid.best_params_)
print(cv_grid.best_estimator_)
print(cv_grid.best_score_)
# Apply it to the test
y_pred = cv_grid.predict(X_test)
# Get confusion matrix and report
print("Classification results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compare it with a DummyClassifier
pipeline.set_params(clf = DummyClassifier())
pipeline.fit(X_train, y_train)
print('-----------------------------------')
print("Dummy results:")
y_dmy = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_dmy))
print(classification_report(y_test, y_dmy))
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
