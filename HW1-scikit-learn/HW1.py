from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

#load data from dataset
X,y = load_breast_cancer(return_X_y=True, as_frame=True)
train, test = train_test_split(X, test_size=0.2, random_state=2020)
train_y, test_y = train_test_split(y, test_size=0.2, random_state=2020)

#print matrix
def matrix(test_y,test_out):
    print(f"{'accuracy:':>10}",accuracy_score(test_y, test_out))
    print(f"{'precision:':>10}",precision_score(test_y, test_out))
    print(f"{'recall:':>10}",recall_score(test_y, test_out))
    print('confusion_matrix\n',confusion_matrix(test_y, test_out))
    precision, recall, _ = precision_recall_curve(test_y, test_out)
    pyplot.plot(recall, precision, marker='.', label='Logistic')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    pyplot.show()
    print(f"{'auc_pr:':>10}",auc(recall, precision))
    fpr, tpr, _ = roc_curve(test_y, test_out)
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    pyplot.show()
    print(f"{'auc:':>10}",roc_auc_score(test_y, test_out))

    
###K means###  
for k in range(1,21):
    print("\n\nK = ",k)
    KNN = KNeighborsClassifier(k)
    KNN.fit(train, train_y)
    test_out = KNN.predict(test)
    matrix(test_y, test_out)

###Decision tree classifier###
from sklearn import tree
for d in range(1,12):
    print("\n\ndepth = ",d)
    DT = tree.DecisionTreeClassifier(max_depth = d).fit(train, train_y)
    test_out = DT.predict(test)
    matrix(test_y, test_out)

###Logistic regression###
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0).fit(train, train_y)
test_out = LR.predict(test)
matrix(test_y, test_out)

###Support vector classifier###
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
SV = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(train, train_y)
test_out = SV.predict(test)
matrix(test_y, test_out)
