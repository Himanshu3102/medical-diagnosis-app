# Diagnosis App
This Python program uses a decision tree algorithm to diagnose diseases based on user inputted symptoms. The program reads in a CSV file of symptom and disease data and utilizes the sklearn library to split the data for training and testing purposes.

The streamlit library is used to create a user interface where the user selects which symptoms they are experiencing from a list of checkboxes. When the "Diagnose" button is pressed, the program uses the decision tree algorithm to predict the disease based on the selected symptoms.

The decision tree algorithm implemented in the program is composed of several functions:

`entropy` calculates the entropy of the data using the mathematical definition of entropy

`best_feature_for_split` finds the best feature for a split by calculating the information gain for each feature and selecting the feature with the highest information gain

`potential_leaf_node` identifies potential leaf nodes in the decision tree
`classify` takes in a decision tree, the feature labels, and a data point, and returns a prediction for the data point
`create_tree` recursively creates a decision tree based on the data and feature labels
The decision tree is then used to predict the disease based on the user inputted symptoms. The program outputs the predicted disease and a message advising the user to consult a doctor.

To run the program, ensure that the required libraries (pandas, sklearn, streamlit) are installed and run the Python script as streamlit `run 1c_manual.py`.
