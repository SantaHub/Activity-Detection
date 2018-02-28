#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:55:23 2017

@author: austin
"""
#Inside Docker deb /home/ActivityDetect

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

def stratified_cv(X, y, clf_class, clf_name, shuffle=True, n_folds=10, **kwargs):
    from sklearn import cross_validation
    import time
    from sklearn.metrics import accuracy_score
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle) #it will have each folds with index of y. Use this index to get corresponding x values 
    clf = OneVsRestClassifier(clf_class(**kwargs),n_jobs=-1) #n_job for number of CPU to be used. -1 for all.
    y_pred = y.copy()
    start_time=time.time()
    #Iterate throught the folds. we will have ii part with 90% of train data index and jj part with 10% of test data index.
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    accuracy = accuracy_score(y_pred,y)*100
    print(str(clf_name)+"\t" +"%.3f"%(time.time()-start_time)+"\t"+ str(accuracy))
    return {'name':clf_name,'y_pred':y_pred ,'acc':accuracy}

def plot_cm(clf, labels = ['b1', 'b2', 'cycl', 'ly', 'sit', 'stand', 'walk']) :
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(y, clf['y_pred'])
    percent = (cm*100.0)/np.array(np.matrix(cm.sum(axis=1)).T) 
    print ('\nConfusion Matrix Stats : ',clf['name'],clf['acc'],'%')
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            print ("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    cax = ax.matshow(percent, cmap=plt.cm.Blues)
    pylab.title('Confusion matrix of the classifier : '+ clf['name']+'\n')
    fig.colorbar(cax)
    ax.set_xticklabels([' '] + labels)
    ax.set_yticklabels([' '] + labels)
    for x,Y in enumerate(percent):
        for a,b in enumerate(Y):
            ax.text(x,a,b,va='center',ha='center',bbox=dict(fc='w',boxstyle='round,pad=1'))
            
    pylab.xlabel('Predicted')
    pylab.ylabel('Actual')
    pylab.savefig('./'+clf['name']+'.png')
    pylab.show()



import pandas as pd
all_data = pd.read_csv('data.csv')

activities = set(all_data['activity'])

# Print number of instances for each activity
for i in activities:
    print( len ( all_data.loc[all_data['activity']==i] ) )

data=(all_data.loc[all_data['activity']=='ly'][:2880])
data=data.append(all_data.loc[all_data['activity']=='walk'][:2880])
data=data.append(all_data.loc[all_data['activity']=='stand'][:2880])
data=data.append(all_data.loc[all_data['activity']=='sit'][:2880])
data=data.append(all_data.loc[all_data['activity']=='cycl'][:2880])
data=data.append(all_data.loc[all_data['activity']=='b2'][:2880])
data=data.append(all_data.loc[all_data['activity']=='b1'][:2880])

data['activity']=data['activity'].replace(activities,list(range(0,len(activities))))

dataX = data.as_matrix().astype(np.float)
datay = data['activity']
y = np.array(datay)


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(dataX) #features made unit varient

from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

GNB=stratified_cv(X, y, naive_bayes.GaussianNB,"Gaussian NB ") #Accuracy 87.66
RFC=stratified_cv(X, y, ensemble.RandomForestClassifier, "Random Forest Classifier",max_depth=4)  #accuracy  99.0347
KNC=stratified_cv(X, y, neighbors.KNeighborsClassifier,"K Neighbors Classifier") #accuracy 94.33
LSV= stratified_cv(X, y, LinearSVC,"Linear Support Vector Classification") #accuracy 97.68
#
##Plot Confusion Matrix
#plot_cm(GNB)
#plot_cm(RFC)
#plot_cm(KNC)
#plot_cm(LSV)
#
#
##Plot Classifier compare
#plot_clf_cmpr([GNB,RFC,KNC,LSV])

