import csv
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing


train_data = []
with open("traindata.csv") as rd:
    data = rd.readlines()
for line in data:
    train_data.append(line.split(","))
train_data = np.array(train_data)
train_data = preprocessing.normalize(train_data)
#print(train_data)


label_data = []
with open("trainlabel.csv") as ld:
    label = ld.readlines()
#label_data.append(line)
label_data = np.array(label)
for i in range(len(label_data)):
    if label_data[i] == '0.0\n':
        label_data[i] = 0
    else:
        label_data[i] = 1
label_data = label_data.astype(np.int32)
#for i in range(len(label_data)):
 #   print(label_data[i])

test_data = []
with open("testdata.csv") as rd:
    data = rd.readlines()
for line in data:
    test_data.append(line.split(","))
test_data = np.array(test_data)
test_data = preprocessing.normalize(test_data)
#print(test_data)


binary_target = np.zeros(len(label_data))
# 5 Fold Cross Validation
kf = KFold(n=len(binary_target), n_folds=5, shuffle=True)

cv = 0
for tr, tst in kf:
    #Train Test Split
    tr_features = []
    tr_target = []
    for i in tr:
        tr_features.append(train_data[i])
        tr_target.append(label_data[i])

    tst_features = []
    tst_target = []
    for i in tst:
        tst_features.append(train_data[i])
        tst_target.append(label_data[i])

    # Training Logistic Regression
    model = LogisticRegression()
    model.fit(train_data, label_data)
    test = model.predict(test_data)
    print(test)
    #Training SVM Model
    #model = SVC() #---------------------#
    #model.fit(train_data, label_data)

    #Training RF Model
    #model = RandomForestClassifier(n_estimators=25)
    #model.fit(train_data, label_data)

    #Training GBDT Model
    #model = GradientBoostingClassifier(n_estimators=200)
    #model.fit(train_data, label_data)

    #Training  Decison Tree
    #model = tree.DecisionTreeClassifier()
    #model.fit(train_data, label_data)


    # Measuring training and test accuracy
    #tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
    #tst_accuracy = np.mean(model.predict(tst_features) == tst_target)
    #print(cv, "Fold Train Accuracy:",tr_accuracy, "Test Accuracy:",tst_accuracy)
    #cv += 1
# print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (
    #    cv, tr_accuracy, tst_accuracy)
#cv += 1

#learn how to build deep learning models to do classification
