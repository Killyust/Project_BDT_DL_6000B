import csv
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

#read training data
train_data = []
with open("traindata.csv") as rd:
    data = rd.readlines()
for line in data:
    train_data.append(line.split(","))
train_data = np.array(train_data)
train_data = preprocessing.normalize(train_data)
#print(train_data)

#read label data
label_data = []
with open("trainlabel.csv") as ld:
    label = ld.readlines()
label_data = np.array(label)
for i in range(len(label_data)):
    if label_data[i] == '0.0\n':
        label_data[i] = 0
    else:
        label_data[i] = 1
label_data = label_data.astype(np.int32)
#for i in range(len(label_data)):
 #   print(label_data[i])

#read test data
test_data = []
with open("testdata.csv") as rd:
    data = rd.readlines()
for line in data:
    test_data.append(line.split(","))
test_data = np.array(test_data)
test_data = preprocessing.normalize(test_data)
#print(test_data)

# Training SVM Model
#model = SVC() #---------------------#
#model.fit(train_data, label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("The accuracy of SVM model is ",acc_t)

#training LogisticRegression model
#model = LogisticRegression()
#model.fit(train_data, label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("The accuracy of LogisticRegression model is ",acc_t)

#Training BaggingClassifier
#model = BaggingClassifier(KNeighborsClassifier(),max_samples = 0.5, max_features = 0.5)
#model.fit(train_data,label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("The accuracy of BaggingClassifier model is ",acc_t)

#Training GBDT Model
#model = GradientBoostingClassifier(n_estimators=200)
#model.fit(train_data, label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("The accuracy of GBDT model is ",acc_t)

#Training  Decision Tree
#model = tree.DecisionTreeClassifier()
#model.fit(train_data, label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("The accuracy of Decision Tree model is ",acc_t)

#Training RF Model
#model = RandomForestClassifier(n_estimators=25)
#model.fit(train_data, label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("The accuracy of RandomForestClassifier model is ",acc_t)

#Training MLPClassfier model
#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,5), random_state=1)
#model.fit(train_data, label_data)
#acc_t = model.score(train_data, label_data, sample_weight=None)
#print("the training accuracy of MLPClassifier model is ",acc_t)

#Training ensemble voting model
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = tree.DecisionTreeClassifier()
model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[1,1,2])
model.fit(train_data, label_data)
acc_t = model.score(train_data, label_data, sample_weight=None)
print("the training accuracy of ensemble voting model is ",acc_t)


test = model.predict(test_data)
test = test.astype(np.int32)

#print(test)
pr = model.predict_proba(test_data)
print(pr)
np.savetxt('project1_2045931.csv', test, delimiter = ',')
