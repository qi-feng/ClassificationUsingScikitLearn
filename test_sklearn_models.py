#QF 2015-03-31
#This practice program is written following scikit-learn website

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
import sys
import numpy as np

#iris = datasets.load_iris()
digits = datasets.load_digits()

#print digits.data.shape

n_samples = len(digits.images)
#create a SVM classifier
clf_svm = svm.SVC(gamma=0.001, C=100., probability=True)

#create a RF classifier
clf_rf = RandomForestClassifier(n_estimators=100)

#create a DT classifier
clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=15) #, max_features="auto")
#clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=10, max_features="auto")

#create AdaBoost classifier
clf_ada = AdaBoostClassifier(n_estimators=100)
clf_bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1.)
clf_bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

#create gradient boosting
clf_gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

#create nearest centroid
clf_nn = NearestCentroid()

#create a stochastic gradient descent classifier
clf_sgd = SGDClassifier(loss="modified_huber", penalty="l2")

#define a training sample and train
train_start = 0
train_stop = n_samples / 2
#clf_svm.fit(digits.data[train_start:train_stop], digits.target[train_start:train_stop])
#clf_rf.fit(digits.data[train_start:train_stop], digits.target[train_start:train_stop])
#clf_dt = DecisionTreeClassifier().fit(digits.data[train_start:train_stop], digits.target[train_start:train_stop])

#define a test sample and test
test_start = n_samples / 2
test_stop = n_samples
#expected_test_sample = digits.target[test_start:test_stop]
#svm_predicted_test_sample = clf_svm.predict(digits.data[test_start:test_stop])
#rf_predicted_test_sample = clf_rf.predict(digits.data[test_start:test_stop])
#predict_proba for SVM (SVC)
#svm_pred_proba_test_sample = clf_svm.predict_proba(digits.data[test_start:test_stop])
#predict_proba for RF
#rf_pred_proba_test_sample = clf_rf.predict_proba(digits.data[test_start:test_stop])

#create a list of classifiers:
#clf_list = [clf_svm, clf_rf, clf_dt, clf_bdt_real, clf_bdt_discrete, clf_gbdt, clf_nn, clf_sgd]
#clf_name_list = ["SVC", "RF", "DT", "BDTr", "BDTd", "gBDT", "NN", "SGD"]
clf_list = [clf_svm, clf_rf, clf_dt, clf_bdt_real, clf_bdt_discrete, clf_gbdt, clf_sgd]
clf_name_list = ["SVC", "RF", "DT", "BDTr", "BDTd", "gBDT", "SGD"]

#for clf_name, clf, predicted_test_sample, pred_proba_test_sample in zip(clf_name_list,
#                                                                        clf_list):
#                                                              [svm_predicted_test_sample, rf_predicted_test_sample],
#                                                              [svm_pred_proba_test_sample, rf_pred_proba_test_sample]):

precision = []
recall = []
f1 = []

outbase = 'test_sklearn_7classifier'
foutname = str(outbase)+'train_results.txt'

#open file foutname for writing results
try:
    print 'Opening file stream for writing results.\n'
    outfile = open(foutname, 'a')
except IOError:
    print 'There was an error opening file', foutname
    sys.exit()


for clf_name, clf in zip(clf_name_list, clf_list):
    #train
    clf.fit(digits.data[train_start:train_stop], digits.target[train_start:train_stop])

    #known true tags for the validating sample
    expected_test_sample = digits.target[test_start:test_stop]
    #predict class for the validating sample
    predicted_test_sample = clf.predict(digits.data[test_start:test_stop])
    #predict_proba for the validating sample
    pred_proba_test_sample = clf.predict_proba(digits.data[test_start:test_stop])

    precision.append(metrics.precision_score(expected_test_sample, predicted_test_sample))
    recall.append(metrics.recall_score(expected_test_sample, predicted_test_sample))
    f1.append(metrics.f1_score(expected_test_sample, predicted_test_sample))

    print("####################################################################################################")
    print("Classification report for classifier %s:\n%s\n"
          % (clf, metrics.classification_report(expected_test_sample, predicted_test_sample)))
    print("Confusion matrix for classifier %s:\n%s"
          % (clf_name, metrics.confusion_matrix(expected_test_sample, predicted_test_sample)))
    print ("log_loss for classifier %s is:\n%s"
          % (clf_name, log_loss(expected_test_sample, pred_proba_test_sample)))
    outfile.write("Classification report using entries %s to %s for classifier %s:\n%s\n"
          % (test_start, test_stop, clf, metrics.classification_report(expected_test_sample, predicted_test_sample)))
    outfile.write("Confusion matrix for classifier %s:\n%s\n"
          % (clf_name, metrics.confusion_matrix(expected_test_sample, predicted_test_sample)))
    outfile.write("log_loss for classifier %s is:\n%s\n"
          % (clf_name, log_loss(expected_test_sample, pred_proba_test_sample)))

plot_list = [precision, recall, f1]
Ncol = 3
fig, ax = plt.subplots(1, Ncol,figsize=(12,4))
title_list=['Precision (false positive)','Recall (false negative)','F1 score']
y_list=['Precision (false positive)','Recall (false negative)','F1 score']
x_list=['Classifier','Classifier','Classifier']

listnum = range(len(clf_list))
for i in range(Ncol):
    print clf_name_list[i], title_list[i], plot_list[i]
    #ax[i/Ncol,i%Ncol].plot([1,2,3],[1,2,3])
    #ax[i%Ncol].plot(plot_list[i],'bo')
    #print listnum, plot_list[i]
    ax[i%Ncol].bar(np.array(listnum)-0.25, plot_list[i], width=0.5, color='b', alpha=0.6)
    ax[i%Ncol].set_title(title_list[i])
    ax[i%Ncol].set_xlabel(x_list[i])
    ax[i%Ncol].set_ylabel(y_list[i])
    ax[i%Ncol].set_xticks(listnum)
    ax[i%Ncol].set_xticklabels(clf_name_list,minor=False)

    ax[i%Ncol].set_xlim(-0.5,len(clf_list)-0.5)
    ax[i%Ncol].set_ylim(0.0,1.1)
    #ax[i/Ncol,i%Ncol].set_ylim([0e-1,9e-1])

figoutputfile = str(outbase)+'_scores.pdf'
plt.tight_layout()
plt.savefig(figoutputfile,format='pdf', dpi=1000)
#plt.show()

show_test = False
#show_test = True
n_test = 5
#print digits.images[n_test]
if show_test:
    print "target for the",str(n_test)+"th entry is:", (digits.target[n_test])
    print "prediction for the",str(n_test)+"th entry is:", clf.predict(digits.data[n_test])

    plt.figure(1, figsize=(3, 3))
    plt.imshow(digits.images[n_test], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

#print digits.target.shape
#print(iris.data)