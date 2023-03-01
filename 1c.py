import copy
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn import tree
import pickle
# import graphviz

df= pd.read_csv("dataset.csv")
# print(df)
symptoms=[]
for i in range(1,18):
    for symptom in df['Symptom_'+ str(i)].unique():
        symptoms.append(symptom)
symptoms=set(symptoms)
symptoms=list(symptoms)
symptoms=symptoms[1:]
symptoms.sort()

# print(symptoms)
X=[]
for i in range(4920):
    symptomsbin=[0 for k in range(131)]
    for element in df.iloc[i]:
        for j in range(len(symptoms)):
            if element==symptoms[j]:
                symptomsbin[j]=1
    X.append(symptomsbin)
Y=list(df["Disease"])
Disease= df["Disease"].unique()
# print(Disease)
# print(Y)
X1=copy.deepcopy(X)
Data=[]
for i in range(len(X)):
    X1[i].append(Y[i])
    Data.append(X1[i])
# print(Data)


X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.1, random_state=100)
clf=DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train,Y_train)
# y=clf.predict(X_test)
with open("modelx.pkl","wb") as file:
    pickle.dump(clf,file)
# print(y)
# print("accuracy score is:" ,accuracy_score(Y_test, y))


print(len(symptoms))






























# dot_data= tree.export_graphviz(clf,out_file=None)
# dot_data = tree.export_graphviz(clf, out_file=None,
#                       feature_names=symptoms,
#                       class_names=Disease,
#                       filled=True, rounded=True,
#                       special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render('jraf')