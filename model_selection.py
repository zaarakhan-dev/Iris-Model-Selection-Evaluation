# Import Libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
iris = load_iris()

X= iris.data
y= iris.target

# Dataset Overview
print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)

# Train Test Split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2 , random_state=42)

# model 1
lr = LogisticRegression(max_iter=200)
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression")

print("Accuracy Score:", accuracy_score(y_test, lr_pred))

print("Precision:", precision_score(y_test,lr_pred, average="weighted"))

print("Recall:", recall_score(y_test,lr_pred, average= "weighted"))

print("F1 Score:", f1_score(y_test,lr_pred, average="weighted"))

# model 2
dt= DecisionTreeClassifier(random_state= 42)

dt.fit(X_train,y_train)

dt_pred=dt.predict(X_test)

# Evaluate Decision Tree Classifier
print("Decision Tree Classifier")

print("Accuracy Score:", accuracy_score(y_test,dt_pred))

print("Precision:", precision_score(y_test,dt_pred, average="weighted"))

print("Recall:", recall_score(y_test,dt_pred, average="weighted"))

print("F1 Score:", f1_score(y_test,dt_pred, average="weighted"))

# model 3
rf= RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)

# Evaluate Random Forest
print("RandomForestClassifier")

print("Accuracy Score:", accuracy_score(y_test, rf_pred))

print("Precision:", precision_score(y_test,rf_pred, average="weighted"))

print("Recall:", recall_score(y_test,rf_pred, average="weighted"))

print("F1 Score:", f1_score(y_test,rf_pred, average="weighted"))

# Cross validation
models={
    'Logistic Regression': lr,
    'Decision Tree': dt,
    'Random Forest': rf
}
for name, model in models.items():

    scores=cross_val_score(model, X,y, cv=5)

    print(name)

    print("Cross Validation Accuracy:", scores.mean())

    print()

# Confusion matrix
cm = confusion_matrix(y_test, lr_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# compare models table
results= pd.DataFrame({
    'Model':[
        'Logistic Regression',
        'Decision Tree',
        'Random Forest'
    ],
    'Accuracy':[
        accuracy_score(y_test,lr_pred),
        accuracy_score(y_test,dt_pred),
        accuracy_score(y_test,rf_pred)
    ]
})

results

sns.barplot(x='Model', y='Accuracy', data=results)
plt.title("Model Comparision")
plt.show()

