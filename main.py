# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:57:23 2020

@author: Sedat
"""

import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns


def PrintClassificationReport(classifierName, classification_report):
    print("\n", classifierName, "Classification Report\n\n", classification_report, "\n")


# Training and Testing Data
forrestTrainingData = pd.read_csv('training.csv')
forrestTestData = pd.read_csv('testing.csv')

# Defining variables of data
X_train = forrestTrainingData.iloc[:, 1:-1]
y_train = forrestTrainingData.iloc[:, 0:1]
X_test = forrestTestData.iloc[:, 1:-1]
y_test = forrestTestData.iloc[:, 0:1]

# Classification with Decision Tree Classifier 
forestDT = tree.DecisionTreeClassifier()
forestDT = forestDT.fit(X_train, y_train)

PrintClassificationReport(
    "Decision Tree",
    classification_report(y_test, forestDT.predict(X_test)))

importances = pd.Series(forestDT.feature_importances_, index=X_train.columns)
f, ax = plt.subplots(figsize=(9, 6))
importances = importances.nlargest(26)
importances.plot(kind='barh',
                 title="The order of importance of attributes for the decision tree")
plt.show()
importances = importances.nlargest(6)

X_trainSpecs = X_train.loc[:, list(importances.index)]
X_testSpecs = X_test.loc[:, list(importances.index)]

# Classification with new attributes 
forestDT = forestDT.fit(X_trainSpecs, y_train)

PrintClassificationReport(
    "2nd Decision Tree",
    classification_report(y_test, forestDT.predict(X_testSpecs)))

corrMatrix = X_trainSpecs.corr()
f, ax = plt.subplots(figsize=(9, 6))
cmap = sns.light_palette("#2ecc71", as_cmap=True)
sns.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax,
            cmap=cmap, vmin=-1, vmax=1)
plt.show()
