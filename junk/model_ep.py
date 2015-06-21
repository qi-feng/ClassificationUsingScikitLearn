__author__ = 'qfeng, epeng'

import numpy as np
import datetime
import pandas as pd
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import log_loss
from scipy.optimize import minimize
import os

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import identity
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score, roc_curve, auc

def str2sec(t):
    #input should have format YYYY-MM-DDTHH:MM:SS
    y, m, d = map(int, t.split('T')[0].split('-'))
    hh, mm, ss = map(int, t.split('T')[1].split(':'))
    t = datetime.datetime(y, m, d, hh, mm, ss)
    return (t-datetime.datetime(1970,1,1)).total_seconds()

def getStartTime(df,category):
    course_start = {}
    df1 = df[df.category == category]
    for key, val in zip(df1.course_id, df1.start):
        if val == 'null':
            continue
        course_start[key] = str2sec(val)
    return course_start

def makeFeature(test=False, outfile='train_xy2.csv'):
    if test==False:
        log_train_df = pd.read_csv('train/log_train.csv')
        train_df = pd.read_csv('train/truth_train.csv', header=None)
        enrollment_train_df = pd.read_csv('train/enrollment_train.csv')
        train_y = train_df[1].values
    else:
        log_train_df = pd.read_csv('test/log_test.csv')
        enrollment_train_df = pd.read_csv('test/enrollment_test.csv')

    enrollmentId = log_train_df['enrollment_id'].unique()
    n = len(enrollmentId)
    rowNum = {enrollmentId[i]: i for i in xrange(n)}
    eventCount = log_train_df.groupby(['enrollment_id', 'event']).size()
    #events = log_train_df['event'].unique()
    events = ['nagivate', 'access', 'problem', 'page_close', 'video','discussion', 'wiki']
    columnName = [s+'_count' for s in events]
    objectCount = pd.DataFrame(np.zeros((n,len(events))), columns=columnName)

    #count number of events for each type of events
    for i in xrange(n):
        for k in eventCount[enrollmentId[i]].keys():
            objectCount[k+'_count'][i] = eventCount[(enrollmentId[i],k)]

    train_x = pd.concat([enrollment_train_df, objectCount], axis=1)

    if test==False:
        data = pd.concat([train_x, pd.DataFrame({'y':train_y})], axis=1)
        data.to_csv(outfile, index=False)
    else:
        data = train_x
        data.to_csv(outfile, index=False)

def feature1(test):
    if test==False:
        log_train_df = pd.read_csv('train/log_train.csv')
    else:
        log_train_df = pd.read_csv('test/log_test.csv')
    obj_df = pd.read_csv('object.csv').drop_duplicates()
    enrollmentId = log_train_df['enrollment_id'].unique()
    n = len(enrollmentId)
    rowNum = {enrollmentId[i]: i for i in xrange(n)}
    timeInSec = [str2sec(s) for s in log_train_df.time]
    log_train_df['time'] = timeInSec
    t0 = 10*24*3600
    tmin = -24*3600
    objectCount = pd.DataFrame(np.zeros((n,2))-t0, columns=['first_problem_time','first_video_time'])
    course_start=getStartTime(obj_df,'course')
    for evt in ['problem','video']:
        df1 = log_train_df[log_train_df['event']==evt].groupby('enrollment_id')['time'].idxmin()
        for k in df1.keys():
            t = log_train_df['time'][df1[k]]
            obj = log_train_df['object'][df1[k]]
            courses = obj_df[obj_df['module_id']==obj].course_id.values
            if len(courses) == 0:
                continue
            if course_start.get(courses[0]) is not None:
                dt = t - course_start[courses[0]]
                if dt < tmin:
                    print dt, courses[0], obj, k, evt, df1[k]
                else:
                    objectCount['first_'+evt+'_time'][rowNum[k]] = dt
            else:
                print 'no course', courses[0]
            if k % 10000 == 0:
                print k, dt
    return objectCount

def addFeature(featureFunc, test=False, infile='train_x.csv', outfile='train_x2.csv'):
    x_df = pd.read_csv(infile)
    objectCount = featureFunc(test)
    data = pd.concat([x_df, objectCount], axis=1)
    data.to_csv(outfile, index=False)


def loadData(inputFile,test=False,sd=1234,scaler=None,testSize=0.1):
    df = pd.read_csv(inputFile)
    course = pd.get_dummies(df.course_id, prefix='course')
    dropCol = ['enrollment_id','username','course_id']
    if not test:
        y = df['y'].values.astype(np.int32)
        dropCol.append('y')
    else:
        enrollmentId = df['enrollment_id']
    df = df.drop(dropCol,axis=1)
    X = np.hstack((course, df.values))
    if not test:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=sd)
        return xTrain, xTest, yTrain, yTest, scaler
    else:
        X = scaler.transform(X).astype(np.float32)
        return X, enrollmentId



def do_RF(train_x, train_y, test_x, test_y, n_estimators=2000, max_depth=20, max_features=20,
          criterion='entropy', method='isotonic', cv=5,
          min_samples_leaf=1, min_samples_split=13, random_state=4141, n_jobs=-1, load=False, save=True,
          outfile=None, search=False):
    if search == False:
        #mdl_name = 'rf_train_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '_maxfeat' + str(max_features) \
        mdl_name = 'rf_isotonic_train_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '_maxfeat' + str(max_features) \
                   + '_minSamLeaf' + str(min_samples_leaf) + '_minSamSplit' + str(min_samples_split) + '.pkl'
        if os.path.exists(mdl_name) == True:
            clf_rf_isotonic = joblib.load(mdl_name)
        else:
            clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                     max_features=max_features, criterion=criterion,
                                                     min_samples_leaf=min_samples_leaf,
                                                     min_samples_split=min_samples_split, random_state=random_state,
                                                     n_jobs=n_jobs)
            clf_rf_isotonic = CalibratedClassifierCV(clf_rf, cv=cv, method=method)
            clf_rf_isotonic.fit(train_x, train_y)
            if save == True:
                try:
                    _ = joblib.dump(clf_rf_isotonic, mdl_name, compress=1)
                except:
                    print("*** Save RF model to pickle failed!!!")
                    if outfile != None:
                        outfile.write("*** Save RF model to pickle failed!!!")
        if test_x != None and test_y != None:
            probas_rf = clf_rf_isotonic.predict_proba(test_x)[:, 1]
            score_rf = roc_auc_score(test_y, probas_rf)
            print("RF ROC score", score_rf)
        return clf_rf_isotonic
    else:
        if test_x is None or test_y is None:
            print "Have to provide test_x and test_y to do grid search!"
            return -1

        min_samples_split = [10, 11, 12]
        max_depth_list = [15, 20, 25]
        n_list = [2000]
        max_feat_list = [10, 20, 30]
        info = {}
        for mss in min_samples_split:
            for max_depth in max_depth_list:
                #for n in n_list:
                for max_features in max_feat_list:
                    print 'max_features = ', max_features
                    n=2000
                    print 'n = ', n
                    print 'min_samples_split = ', mss
                    print 'max_depth = ', max_depth
                    clf_rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, max_features=max_features,
                                                    criterion=criterion, min_samples_leaf=min_samples_leaf,
                                                    min_samples_split=mss, random_state=random_state, n_jobs=n_jobs)
                    #clf_rf.fit(train_x, train_y)
                    clf_rf_isotonic = CalibratedClassifierCV(clf_rf, cv=cv, method=method)
                    clf_rf_isotonic.fit(train_x, train_y)
                    probas_rf = clf_rf_isotonic.predict_proba(test_x)[:, 1]
                    scores = roc_auc_score(test_y, probas_rf)
                    info[max_features, mss, max_depth] = scores
        for mss in info:
            scores = info[mss]
            print(
                'clf_rf_isotonic: max_features = %d, min_samples_split = %d, max_depth = %d, ROC score = %.5f(%.5f)' % (mss[0], mss[1], mss[2], scores.mean(), scores.std()))


def do_gbdt(train_x, train_y, test_x, test_y, learning_rate=0.03, max_depth=8, max_features=25,
            n_estimators=600, load=False, save=True, outfile=None, search=False):
    if search == False:
        mdl_name = 'gbdt_train_lr' + str(learning_rate) + '_n' + str(n_estimators) + '_maxdep' + str(max_depth) + '.pkl'
        if os.path.exists(mdl_name) == True:
            clf_gbdt = joblib.load(mdl_name)
        else:
            # create gradient boosting
            clf_gbdt = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                                  max_features=max_features, n_estimators=n_estimators)
            #n_estimators=500, learning_rate=0.5, max_depth=3)
            clf_gbdt.fit(train_x, train_y)
            if save == True:
                try:
                    _ = joblib.dump(clf_gbdt, mdl_name, compress=1)
                except:
                    print("*** Save GBM model to pickle failed!!!")
                    if outfile != None:
                        outfile.write("*** Save RF model to pickle failed!!!")
        if test_x != None and test_y != None:
            probas_gbdt = clf_gbdt.predict_proba(test_x)[:, 1]
            score_gbdt = roc_auc_score(test_y, probas_gbdt)
            print("GBDT ROC score", score_gbdt)
        return clf_gbdt
    else:
        max_depth_list = [5,6,7]
        n_list = [2000, 3000]
        lr_list = [0.01, 0.005]
        info = {}
        for md in max_depth_list:
            for n in n_list:
                for lr in lr_list:
                    print 'max_depth = ', md
                    print 'n = ', n
                    print 'learning rate = ', lr
                    clf_gbdt = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=md,
                                                          max_features=max_features, n_estimators=n_estimators)
                    # n_estimators=500, learning_rate=0.5, max_depth=3)
                    clf_gbdt.fit(train_x, train_y)
                    probas_gbdt = clf_gbdt.predict_proba(test_x)[:, 1]
                    score_gbdt = roc_auc_score(test_y, probas_gbdt)
                    info[md, n, lr] = score_gbdt
        for md in info:
            scores = info[md]
            print('GBDT max_depth = %d, n = %d, lr = %.5f, ROC score = %.5f(%.5f)' % (
                md[0], md[1], md[2], scores.mean(), scores.std()))


def do_nn(xTrain, yTrain, test_x, test_y, dropout_in=0.2, dense0_num=600, dropout_p=0.4, dense1_num=1200,
                  update_learning_rate=0.00002,
                  update_momentum=0.9, test_ratio=0.2, max_epochs=40, search=False):
    num_features = len(xTrain[0, :])
    num_classes = 2
    print num_features
    if search == False:
        layers0 = [('input', InputLayer),
                   ('dropoutin', DropoutLayer),
                   ('dense0', DenseLayer),
                   ('dropout', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('output', DenseLayer)]
        clf = NeuralNet(layers=layers0,
                        input_shape=(None, num_features),
                        dropoutin_p=dropout_in,
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
        if test_x is not None and test_y is not None:
            probas_nn = clf.predict_proba(test_x)[:, 1]
            score_nn = roc_auc_score(test_y, probas_nn)
            print("NN ROC score", score_nn)
        return clf
    else:
        dropout_in_list = [0.2]
        dense0_num_list = [500, 1000, 1500]
        dropout_p_list = [0.5]
        dense1_num_list = [400, 800, 1200]
        info = {}
        for d_in in dropout_in_list:
            for d_01 in dropout_p_list:
                for d0 in dense0_num_list:
                    for d1 in dense1_num_list:
                        print 'dropout_in = ', d_in
                        print 'dense0_num = ', d0
                        print 'dropout_p = ', d_01
                        print 'dense0_num = ', d1
                        layers0 = [('input', InputLayer),
                                   ('dropoutin', DropoutLayer),
                                   ('dense0', DenseLayer),
                                   ('dropout', DropoutLayer),
                                   ('dense1', DenseLayer),
                                   ('output', DenseLayer)]
                        clf = NeuralNet(layers=layers0,
                                        input_shape=(None, num_features),
                                        dropoutin_p=dropout_in,
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
                        probas_nn = clf.predict_proba(test_x)[:, 1]
                        score_nn = roc_auc_score(test_y, probas_nn)
                        print("NN ROC score", score_nn)
                        info[d_in, d0, d_01, d1] = score_nn
        for md in info:
            scores = info[md]
            print('NN dropout_in = %.5f, dense0_num = %d, dropout_p = %.5f, dense1_num = %d, ROC score = %.5f(%.5f)' % \
                  (md[0], md[1], md[2], md[3], scores.mean(), scores.std()))

def roc_func(weights, predictions, test_y):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
    return roc_auc_score(test_y, final_prediction)

def make_predictions(clfs, test_file, scaler, test_x, test_y, outfile='test_sub.csv'):
    scores = []
    predictions = []
    for clf in clfs:
        _probas = clf.predict_proba(test_x)[:,1]
        _score = roc_auc_score(test_y, _probas)
        print("ROC score", _score)
        predictions.extend(_probas)
        scores.extend(_score)

    starting_values = [0.5]*len(predictions)

    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(predictions)

    res = minimize(roc_func, starting_values, args= (predictions, test_y), method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    predict_x, enrollment_id = loadData(test_file,test=True,scaler=scaler)
    predict_y=None
    for clf, wt in zip(clfs, res['x']):
        pred_test_current_clf = pd.DataFrame((clf.predict_proba(predict_x))) * wt
        if predict_y==None:
            predict_y = pred_test_current_clf
        else:
            predict_y = predict_y + pred_test_current_clf

    sub = pd.concat([pd.DataFrame(enrollment_id), pd.DataFrame(predict_y)],axis=1)
    sub.to_csv(outfile, index=False, header=False)


def run(train_file='train_x.csv', test_file='test_x.csv', outfile='sub.csv'):
    testRatio = 0.1
    xTrain, xTest, yTrain, yTest, scaler = loadData(train_file,sd=1234,testSize=testRatio)

    clf_nn = do_nn(xTrain, yTrain, xTest, yTest, dropout_in=0.1, dense0_num=200, dropout_p=0.5,
                   dense1_num=400, update_learning_rate=0.00003, update_momentum=0.9, test_ratio=testRatio, max_epochs=50)
    clf_rf = do_RF(xTrain, yTrain, xTest, yTest)
    clf_gbdt = do_gbdt(xTrain, yTrain, xTest, yTest)

    make_predictions([clf_rf, clf_gbdt,clf_nn], test_file, scaler, xTest, yTest,
                     outfile=outfile)


if __name__ == '__main__':
    #makeFeature(test=False, outfile='train_x.csv')
    #addFeature(feature1, test=False, infile='train_x.csv', outfile='train_x2.csv')
    #makeFeature(test=True, outfile='test_x.csv')
    #addFeature(feature1, test=True, infile='test_x.csv', outfile='test_x2.csv')
    #tune(train_file='train_x2.csv')
    run(train_file='train_x2.csv', test_file='test_x2.csv', outfile='sub2.csv')
