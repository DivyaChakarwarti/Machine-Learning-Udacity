
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

"""
   OUTPUT
       no. of Chris training emails: 7936
       no. of Sara training emails: 7884
       No of featues: 3785
       Accuracy: 0.9789533560864618

"""
    
import sys
from time import time
sys.path.append("/home/divya/Downloads/ud120-projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###


#########################################################

print "No of featues: " + str(len(features_train[0]))
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)
y_pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
print "Accuracy: " + str(accuracy_score(labels_test, y_pred))
