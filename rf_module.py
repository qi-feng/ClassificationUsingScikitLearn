import sys
import numpy as np
import cPickle as pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import metrics
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, tanh
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn import cross_validation, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime as dt
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import os.path
from itertools import izip, count
from nn_module import fit_two_dense
from sklearn.calibration import CalibratedClassifierCV


### read from files
def read_no_transform(trainf='train.csv', testf='test.csv', samplef='sampleSubmission.csv', train_ratio = 0.9):
    train = pd.read_csv(trainf)
    test = pd.read_csv(testf)
    sample = pd.read_csv(samplef)

    ### drop ids and labels
    labels = train['target']
    train.drop(['target', 'id'], axis=1, inplace=True)
    test = test.drop('id', axis=1)

    print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

    test_rows = test.shape[0]

    ### encode labels
    #lbl_enc = preprocessing.LabelEncoder()
    #labels = lbl_enc.fit_transform(labels)

    print(train.head())

    test_ratio = 1 - train_ratio

    ### train/test split for cross-validation
    sss = StratifiedShuffleSplit(labels, test_size=test_ratio, random_state=1234)
    for train_index, test_index in sss:
        break

    train_x, train_y = train.values[train_index], labels.values[train_index]
    test_x, test_y = train.values[test_index], labels.values[test_index]
    return train_x, train_y, test_x, test_y, train_index, test_index, test, labels, sample

def do_RF(train_x, train_y, n_estimators=2000, max_depth=45, max_features=40, criterion='entropy',
          min_samples_leaf=1, min_samples_split=5, random_state=4141, n_jobs=-1, load=False, save=True, outfile=None):
    mdl_name = 'rf_train_n'+str(n_estimators)+'_maxdep'+str(max_depth)+'_maxfeat'+str(max_features)\
               +'_minSamLeaf'+str(min_samples_leaf)+'_minSamSplit'+str(min_samples_split)+'.pkl'

    if os.path.exists(mdl_name)==True:
        clf_rf = joblib.load(mdl_name)
    else:
        clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                        criterion=criterion, min_samples_leaf=min_samples_leaf,
                                        min_samples_split=min_samples_split, random_state=random_state, n_jobs=n_jobs)
        clf_rf.fit(train_x, train_y)
        if save==True:
            try:
                _ = joblib.dump(clf_rf, mdl_name,compress=1) 
            except:
                print("*** Save RF model to pickle failed!!!")
                if outfile!=None:
                    outfile.write("*** Save RF model to pickle failed!!!")

    return clf_rf

def do_RF_v1(train_x, train_y, n_estimators=700, max_features=10, random_state=4141, n_jobs=-1,
             method='isotonic', cv=7, load=False, save=True, outfile=None):
    mdl_name = 'clf_isotonic+rf_train_n'+str(n_estimators)+'_maxfeat'+str(max_features)+'.pkl'

    if os.path.exists(mdl_name)==True:
        clf_isotonic = joblib.load(mdl_name)
    else:
        clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                        random_state=random_state, n_jobs=n_jobs)
        clf_isotonic = CalibratedClassifierCV(clf_rf, cv=cv, method=method)

        clf_isotonic.fit(train_x, train_y)
        if save==True:
            try:
                _ = joblib.dump(clf_isotonic, mdl_name,compress=1)
            except:
                print("*** Save RF model to pickle failed!!!")
                if outfile!=None:
                    outfile.write("*** Save RF model to pickle failed!!!")

    return clf_isotonic

def do_gbdt(train_x, train_y, learning_rate=0.03, max_depth=8, max_features=25, n_estimators=600,
            load=False, save=True, outfile=None):
    mdl_name = 'gbdt_train_lr'+str(learning_rate)+'_n'+str(n_estimators)+'_maxdep'+str(max_depth)+'.pkl'

    if os.path.exists(mdl_name)==True:
        clf_gbdt = joblib.load(mdl_name)
    else:
        #create gradient boosting
        clf_gbdt = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth,
                   max_features=max_features, n_estimators=n_estimators)
                                              #n_estimators=500, learning_rate=0.5, max_depth=3)
        clf_gbdt.fit(train_x, train_y)
        if save==True:
            try:
                _ = joblib.dump(clf_gbdt, mdl_name, compress=1)
            except:
                print("*** Save GBM model to pickle failed!!!")
                if outfile!=None:
                    outfile.write("*** Save RF model to pickle failed!!!")

    return clf_gbdt

def do_svc(train_x, train_y, C=3.467, gamma=0.05754, probability=True, load=False, save=True, outfile=None):
    mdl_name = 'svc_no_transform_train_C'+str(C)+'_g'+str(gamma)+'.pkl'
    if os.path.exists(mdl_name)==True:
        clf_svc = joblib.load(mdl_name)
    else:
        clf_svc = svm.SVC(C=C, gamma=gamma, probability=probability)
        clf_svc.fit(train_x, train_y)

        if save==True:
            try:
                _ = joblib.dump(clf_svc, mdl_name, compress=1)
            except:
                print("*** Save SVC model to pickle failed!!!")
                if outfile!=None:
                    outfile.write("*** Save RF model to pickle failed!!!")

    return clf_svc

def do_logreg(train_x, train_y):
    logreg = LogisticRegression()
    logreg.fit(train_x, train_y)
    return logreg

def print_clf(clf, test_x, test_y, clf_name=None, outfname=None):
    print('Classifier {name} has a LogLoss {score}'.format(name=clf_name, score=log_loss(test_y, clf.predict_proba(test_x))))
    if outfname!=None:
        outfname.write('Classifier {name} has a LogLoss {score} \n'.format(name=clf_name, score=log_loss(test_y, clf.predict_proba(test_x))))

def predict_ones(predictions, threshold=0.99):
    #for max_val, max_index in zip(np.amax(predictions, axis=1), predictions.argmax(axis=1)):
    for i, max_val, max_index in izip(count(), np.amax(predictions, axis=1), predictions.argmax(axis=1)):
        if max_val > threshold:
            predictions[i][:]=0.0
            predictions[i][max_index] = 1.0
    return predictions

###### end of functions

if __name__ == '__main__':
    train_ratio = 0.9
    train_x, train_y, test_x, test_y, train_index, test_index, test, labels, sample = read_no_transform(train_ratio=train_ratio)

    foutname = 'otto_ensemble'+str(train_ratio*100)+'percent_training_set_results_with_nn_v10_dropoutinput.txt'
    ### open file foutname for writing results
    try:
        print 'Opening file stream for writing results.\n'
        outfile = open(foutname, 'a')
    except IOError:
        print 'There was an error opening file', foutname
        sys.exit()

    ### log transformed features:
    X, y, encoder, scaler = load_train_data('train.csv')
    train_x2, train_y2 = X[train_index], y[train_index]
    test_x2, test_y2 = X[test_index], y[test_index]

    ### BM25 transfomred features:
    #X3, y3, encoder3, scaler3 = load_train_data('train.csv',transform='BM25')
    #train_x3, train_y3 = X3[train_index], y3[train_index]
    #test_x3, test_y3 = X3[test_index], y3[test_index]
    #train_x3 = 2.0*train_x/(train_x+1.0)
    #train_y3 = train_y
    #test_x3 = 2.0*test_x/(test_x+1.0)
    #test_y3 =


    ### building the classifiers, train them and predict for test set if desired
    clfs = []

    clf_rf = do_RF(train_x, train_y, outfile=outfile)
    #clf_rf = RandomForestClassifier(n_estimators=700, max_depth=10)
    print_clf(clf_rf, test_x, test_y, clf_name="RF", outfname=outfile)
    clfs.append(clf_rf)

    clf_gbdt = do_gbdt(train_x, train_y, outfile=outfile)
    print_clf(clf_gbdt, test_x, test_y, clf_name="GBDT", outfname=outfile)
    clfs.append(clf_gbdt)

    #clf_nn = fit(train_x2, train_y2, dropout_in=0.2, dense0_num=400, dropout_p=0.4, dense1_num=700, dropout1_p=0.5, dense2_num=100, update_learning_rate=0.005, max_epochs=50)
    clf_nn = fit_two_dense(train_x2, train_y2, dropout_in =0.1, dense0_num=800, dropout_p=0.5, dense1_num=500, update_learning_rate=0.007, max_epochs=100)
    print_clf(clf_nn, test_x2, test_y2, clf_name="Neural Network log transformed (cheating test)", outfname=outfile)
    clfs.append(clf_nn)

    clf_nn2 = fit_two_dense(train_x2, train_y2, dropout_in =0.1, dense0_num=800, dropout_p=0.5, dense1_num=500, update_learning_rate=0.01, max_epochs=50)
    print_clf(clf_nn2, test_x2, test_y2, clf_name="Neural Network log transformed (cheating test)", outfname=outfile)
    clfs.append(clf_nn2)

    clf_nn3 = fit_two_dense(train_x2, train_y2, dropout_in =0.1, dense0_num=800, dropout_p=0.5, dense1_num=500, update_learning_rate=0.01, max_epochs=50)
    print_clf(clf_nn3, test_x2, test_y2, clf_name="Neural Network log transformed (cheating test)", outfname=outfile)
    clfs.append(clf_nn3)

    clf_svc = do_svc(train_x, train_y, C=60., gamma=0.00095, outfile=outfile)
    print_clf(clf_svc, test_x, test_y, clf_name="SVC no transform", outfname=outfile)
    clfs.append(clf_svc)

    ### finding the optimum weights
    predictions = []
    for i, clf in enumerate(clfs):
        if i<2:
            predictions.append((clf.predict_proba(test_x)))
        elif i<5:
            predictions.append((clf.predict_proba(test_x2)))
        elif i==5:
            predictions.append((clf.predict_proba(test_x)))

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight*prediction

        return log_loss(test_y, final_prediction)

    #the algorithms need a starting value, right not we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    starting_values = [0.5]*len(predictions)

    #adding constraints  and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)

    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    outfile.write('Ensemble Score: {best_score} \n'.format(best_score=res['fun']))
    outfile.write('Best Weights: {weights} \n'.format(weights=res['x']))

    sub_file = 'otto_test_submit_benchmark_with_nn_v10_dropoutinput.csv'

    #pred_test=pd.DataFrame(np.zeros((test_rows, 9)), index=sample.id.values, columns=sample.columns[1:])
    counter = 0
    #pred_test = []
    for clf, wt in zip(clfs, res['x']):
        #test_outfile.write('Weight for this classifier: {} \n'.format(wt))

        #pred_test_current_clf.to_csv(test_res_file, index_label='id')
        if counter==0:
            pred_test_current_clf = pd.DataFrame((clf.predict_proba(test))) * wt
            pred_test = pred_test_current_clf
        elif counter==1:
            pred_test_current_clf = pd.DataFrame((clf.predict_proba(test))) * wt
            pred_test = pred_test + pred_test_current_clf
        elif counter<5:
            #scaler = StandardScaler()
            #test_X = scaler.fit_transform(np.log(test+1))
            test_X, test_ids = load_test_data('test.csv', scaler)
            pred_test_current_clf = pd.DataFrame((clf.predict_proba(test_X))) * wt
            pred_test = pred_test + pred_test_current_clf
        elif counter==5:
            #scaler = StandardScaler()
            #test_X = scaler.fit_transform(2.0*test/(test+1))
            #test_X = 2.0*test/(test+1.0)
            pred_test_current_clf = pd.DataFrame((clf.predict_proba(test))) * wt
            pred_test = pred_test + pred_test_current_clf

        counter += 1

    pred_test.columns = sample.columns[1:]
    pred_test.index += 1
    pred_test.to_csv(sub_file, index_label='id')

    print("\nJob done in {}\n".format(dt.now() - t0))
    outfile.write("Job done in {}\n".format(dt.now() - t0))

    outfile.close()


