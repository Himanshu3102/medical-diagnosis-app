import operator
from collections import Counter
import math
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("Diagnosis App")
df= pd.read_csv("dataset.csv")#getting data out of csv file into the dataframe

symptoms=[]
for i in range(1,18):
    for symptom in df['Symptom_'+ str(i)].unique():
        symptoms.append(symptom)
symptoms=set(symptoms)
#The symptoms are added to a set to get a list of all the differnt symptoms present.
symptoms=list(symptoms)
# print(symptoms)
symptoms=symptoms[1:]
# print(symptoms)
symptoms=[x for x in symptoms if type(x)==str]
symptoms.sort()
inp=[0 for i in range(len(symptoms))]
for i in range(len(inp)):
    if st.checkbox(symptoms[i]):
        inp[i]=1
#The above part of the code is used to get a list of the symptoms from the user which he/she is experiencing. Streamlit is being used to the get the data from the checkboxes.
if st.button("Diagnose"): #As soom as the button is pressed a boolean vector is generated to be matched in the dataset.
    X = []
    for i in range(4920):
        symptomsbin = [0 for k in range(131)]
        for element in df.iloc[i]:
            for j in range(len(symptoms)):
                if element == symptoms[j]:
                    symptomsbin[j] = 1
        X.append(symptomsbin)
    Y = list(df["Disease"])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=100)
    #have used the sklearn library to just split the data. We have implemented the decision tree on our own !

    X1 = copy.deepcopy(X_train)
    Data = []
    for i in range(len(X1)):
        X1[i].append(Y_train[i])
        Data.append(X1[i])


    # Mathematical definition of entropy
    def entropy(data):
        frequency = Counter([item[-1] for item in data])

        def item_entropy(category):
            ratio = float(category) / len(data)
            return -1 * ratio * math.log(ratio, 2)

        return sum(item_entropy(c) for c in frequency.values())

        #The function gets the best feature for split. e(v) stores the entropy times proportion of one of the sides of the split by the feature f.
        #This is done for each feature and the corresponding information gain is calculated and stored.
        #The feature which gives the highest information gain is used to split the data.
    def best_feature_for_split(data):
        baseline = entropy(data)

        def feature_entropy(f):
            def e(v):
                partitioned_data = [d for d in data if d[f] == v]
                proportion = (float(len(partitioned_data)) / float(len(data)))
                return proportion * entropy(partitioned_data)

            return sum(e(v) for v in set([d[f] for d in data]))

        features = len(data[0]) - 1
        information_gain = [baseline - feature_entropy(f) for f in range(features)]
        best_feature, best_gain = max(enumerate(information_gain), key=operator.itemgetter(1))
        return best_feature


    def potential_leaf_node(data):
        count = Counter([i[-1] for i in data])
        return count.most_common(1)[0]


    def classify(tree, label, data):
        root = list(tree.keys())[0]
        node = tree[root]
        index = label.index(root)
        for k in node.keys():
            if data[index] == k:
                if isinstance(node[k], dict):
                    return classify(node[k], label, data)
                else:
                    return node[k]


    def create_tree(data, label):
        category, count = potential_leaf_node(data)
        if count == len(data):
            return category
        node = {}
        feature = best_feature_for_split(data)
        feature_label = label[feature]
        node[feature_label] = {}
        classes = set([d[feature] for d in data])
        for c in classes:
            partitioned_data = [d for d in data if d[feature] == c]
            node[feature_label][c] = create_tree(partitioned_data, label)
        return node

    # From the above functions, a tree is created.
    #Now, we use the classify function to get the prediction.
    #Here we have just calculated the accuracy but one can make modifications to the code to get other metrics as well.
    symptoms.append("Disease")
    tree = create_tree(Data, symptoms)
    print(tree)
    score = 0
    for i in range(len(X_test)):
        if (Y_test[i] == classify(tree, symptoms, X_test[i])):
            score += 1
        # print("label: ", Y_test[i], "predicted: ", classify(tree, symptoms, X_test[i]))
    print("accuracy= ", score / len(Y_test))
    print(classify(tree, symptoms, inp))
    # st.write("working on it.....")
    # st.write("wait some time.....")
    st.write("You are diagnosed with : ",classify(tree, symptoms, inp))
    st.write("Please consult a doctor")
