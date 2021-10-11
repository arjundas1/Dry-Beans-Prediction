import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, svm, metrics

data = pd.read_csv("C:/Users/Arjun/PycharmProjects/DataScience/Mutiple Regression/Dry Bean.csv", sep=",")
print(data.head())
le = preprocessing.LabelEncoder()
cls = le.fit_transform(list(data["Class"]))
y = list(cls)
x = data[["Perimeter", "MajorAxisLength", "MinorAxisLength", "EquivDiameter", 
          "Solidity", "roundness", "ShapeFactor1", "ShapeFactor2", 
          "ShapeFactor3", "ShapeFactor4"]]

knnbest = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    knnmodel = KNeighborsClassifier(n_neighbors=3)
    knnmodel.fit(x_train, y_train)
    knnacc = knnmodel.score(x_test, y_test)
    if knnacc > knnbest:
        knnbest = knnacc
        with open("KNNBeans.pickle", "wb") as f:
            pickle.dump(knnmodel, f)


Knnpickle = open("KNNBeans.pickle", "rb")
knnmodel = pickle.load(Knnpickle)
knnacc = knnmodel.score(x_test, y_test)
print("Accuracy using KNN: ", round(knnacc*100, 2), "%")

svmbest = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    svmmodel = svm.SVC(kernel="linear", C=1)
    svmmodel.fit(x_train, y_train)
    y_pred = svmmodel.predict(x_test)
    svmacc = metrics.accuracy_score(y_test, y_pred)
    if svmacc > svmbest:
        svmbest = svmacc
        with open("SVMBeans.pickle", "wb") as f:
            pickle.dump(svmmodel, f)


Svmpickle = open("SVMBeans.pickle", "rb")
svmmodel = pickle.load(Svmpickle)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy using SVM: ", round(svmacc*100, 2), "%")

lrbest = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    lr = LogisticRegression(solver="liblinear")
    lrmodel = lr.fit(x_train, y_train)
    lracc = lr.score(x_test, y_test)
    if lracc > lrbest:
        lrbest = lracc
        with open("LoReBeans.pickle", "wb") as f:
            pickle.dump(lrmodel, f)


Lrpickle = open("SVMBeans.pickle", "rb")
lrmodel = pickle.load(Lrpickle)
lracc = lr.score(x_test, y_test)
print("Accuracy using Logistic Regression: ", round(lracc*100, 2), "%")
