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
from lasagne.layers import Conv1DLayer
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

def load_train_data(path, ordered=True, transform='log'):
    df = pd.read_csv(path)
    X = df.values.copy()
    if ordered==False:
        np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    if transform==None:
        X = scaler.fit_transform(X)              #0.483907758842
    if transform=='log':
        X = scaler.fit_transform(np.log(X+1))     #0.463801294038
    if transform=='BM25':
        X = scaler.fit_transform(2.0*X/(X+1.0))    #0.492644853714
    #X = scaler.fit_transform(np.log(np.log(X+1)+1)) #0.477841919735
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(np.log(X+1))
    return X, ids


def fit(xTrain, yTrain, dropout_in=0.2, dense0_num=800, dropout_p=0.5, dense1_num=500, update_learning_rate=0.01,
        dense2_num=500, dropout1_p=0.5,
        update_momentum=0.9, test_ratio=0.2, max_epochs=20):

    #xTrain, yTrain, encoder, scaler = load_train_data(train_fname)
    #xTest, ids = load_test_data('test.csv', scaler)
   #xTrain, yTrain, encoder, scaler = load_train_data(train_fname)
    #xTest, ids = load_test_data('test.csv', scaler)

    num_features = len(xTrain[0,:])
    num_classes = 9
    print num_features

    layers0 = [('input', InputLayer),
           #('conv1', Conv1DLayer),
           #('dropout_in', DropoutLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]


    clf = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 #conv1_num_filters=32, 
                 #conv1_filter_size=9,
                 #conv1_border_mode='same',
                 dropoutf_p = dropout_in,
                 dense0_num_units=dense0_num,
                 dropout_p=dropout_p,
                 dense1_num_units=dense1_num,
                 dropout1_p=dropout1_p,
                 dense2_num_units=dense2_num,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=nesterov_momentum,
                 update_learning_rate=update_learning_rate,
                 update_momentum=update_momentum,
                 eval_size=test_ratio,
                 verbose=1,
                 max_epochs=max_epochs)

    clf.fit(xTrain, yTrain)
    ll_train = metrics.log_loss(yTrain, clf.predict_proba(xTrain))
    print ll_train

    return clf

def fit_two_dense(xTrain, yTrain, dropout_in=0.2, dense0_num=800, dropout_p=0.5, dense1_num=500, update_learning_rate=0.008,
        update_momentum=0.9, test_ratio=0.2, max_epochs=40):

    #xTrain, yTrain, encoder, scaler = load_train_data(train_fname)
    #xTest, ids = load_test_data('test.csv', scaler)
   #xTrain, yTrain, encoder, scaler = load_train_data(train_fname)
    #xTest, ids = load_test_data('test.csv', scaler)

    num_features = len(xTrain[0,:])
    num_classes = 9
    print num_features

    layers0 = [('input', InputLayer),
           ('dropoutin', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]


    clf = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 dropoutin_p = dropout_in,
                 dense0_num_units=dense0_num,
                 dropout_p=dropout_p,
                 dense1_num_units=dense1_num,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=nesterov_momentum,
                 update_learning_rate=update_learning_rate,
                 update_momentum=update_momentum,
                 eval_size=test_ratio,
                 verbose=1,
                 max_epochs=max_epochs)

    clf.fit(xTrain, yTrain)
    ll_train = metrics.log_loss(yTrain, clf.predict_proba(xTrain))
    print ll_train

    return clf

if __name__ == '__main__':
    train_ratio = 0.9
    test_ratio = 1 - train_ratio

    train = pd.read_csv('train.csv')
    labels = train['target']

    sss = StratifiedShuffleSplit(labels, test_size=test_ratio, random_state=1234)
    for train_index, test_index in sss:        
        break

    X, y, encoder, scaler = load_train_data('train.csv')
    train_x2, train_y2 = X[train_index], y[train_index]
    test_x2, test_y2 = X[test_index], y[test_index]
    
    #clf_nn = fit(train_x2, train_y2, dropout_in=0.2, dense0_num=400, dropout_p=0.4, dense1_num=700, dropout1_p=0.5, dense2_num=100, update_learning_rate=0.005, max_epochs=50)
    clf_nn = fit_twolayer(train_x2, train_y2, dropin=0.2, dense0_num=800, dropout_p=0.5, dense1_num=500, update_learning_rate=0.005, max_epochs=50)

    def print_clf(clf, test_x, test_y, clf_name=None, outfname=None):
        print('Classifier {name} has a LogLoss {score}'.format(name=clf_name, score=log_loss(test_y, clf.predict_proba(test_x))))
        if outfname!=None:
            outfname.write('Classifier {name} has a LogLoss {score} \n'.format(name=clf_name, score=log_loss(test_y, clf.predict_proba(test_x))))

    print_clf(clf_nn, test_x2, test_y2, clf_name="Neural Network log transformed (cheating test)", outfname=outfile)


