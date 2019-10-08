
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
'''
OUTPUT
    no. of Chris training emails: 7936
    no. of Sara training emails: 7884
    Accuracy: 0.9908987485779295
    1
    0
    1
    877
'''

import sys
from time import time
sys.path.append("/home/divya/Downloads/ud120-projects/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

'''

clf = svm.SVC(kernel='linear')
clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
print "Accuracy: " + str(accuracy_score(labels_test, pred))
'''
from sklearn import svm
clf = svm.SVC(C=10000.0,kernel='rbf')
#clf = svm.SVC(kernel='linear')
clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
print "Accuracy: " + str(accuracy_score(labels_test, pred))
print pred[10]
print pred[26]
print pred[50]

count=0
for i in range (1,len(pred)):
    if pred[i]==1:
        count+=1
print count 
