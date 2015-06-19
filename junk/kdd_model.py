__author__ = 'qfeng'

import numpy as np
import datetime
import matplotlib.pyplot as plt
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
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score, roc_curve, auc

def iso2sec(t):
    #input should have format YYYY-MM-DDTHH:MM:SS
    y, m, d = t.split('T')[0].split('-')
    h, mt, s = t.split('T')[1].split(':')
    dt = datetime.datetime(int(y), int(m), int(d), int(h), int(mt), int(s))
    return dt

def diff_sec(t1, t2):
    #return t2-t1 in unit of sec
    #input should have format YYYY-MM-DDTHH:MM:SS
    dt1 = iso2sec(t1)
    dt2 = iso2sec(t2)
    return (dt2-dt1).total_seconds()

def make_train():
    log_train_df = pd.read_csv('train/log_train.csv')
    train_df = pd.read_csv('train/truth_train.csv', header=None)
    enrollment_train_df = pd.read_csv('train/enrollment_train.csv')
    obj_df = pd.read_csv('object.csv')

    train_y = train_df[1].values

    event_count = log_train_df.groupby('enrollment_id', as_index=False).count()
    event_count = event_count.rename(columns = {'time':'event_count'})

    object_count = event_count.copy()
    object_count['access_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['problem_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['page_close_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['video_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['nagivate_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['discussion_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['wiki_count'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)

    object_count['first_problem_time'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)
    object_count['first_video_time'] = pd.Series(np.zeros(len(object_count)), index=object_count.index)

    course_start = obj_df.start[(obj_df.start != 'null') & (obj_df.category=='course')]
    chapter_start = obj_df.start[(obj_df.start != 'null') & (obj_df.category=='chapter')]

    #print log_train_df['event'].unique()
    for i, en_id in enumerate(object_count['enrollment_id']):
        #count number of events for each type of events
        res = log_train_df[(log_train_df['enrollment_id']==en_id)]['event'].value_counts()
        for k, val in zip(res.keys(), res.values):
            #object_count[k+'_count'][i] = val
            object_count.loc[i,(k+'_count')] = val
        #get the earliest time that a problem and a video that is accessed
        t_first_problem = log_train_df[(log_train_df['enrollment_id']==en_id) & (log_train_df['event']=='problem')].time.values[0]
        t_first_video = log_train_df[(log_train_df['enrollment_id']==en_id) & (log_train_df['event']=='video')].time.values[0]
        t_first_problem = diff_sec(course_start, t_first_problem)
        t_first_video = diff_sec(course_start, t_first_video)
        object_count['first_problem_time'][i] = t_first_problem
        object_count['first_video_time'][i] = t_first_video
        if i%1000 == 1:
            print(str(i)+"th enrollment processed")
            print object_count.iloc(i-1)


    object_count.drop('enrollment_id', axis=1, inplace=True)
    train_x = pd.concat([enrollment_train_df, object_count], axis=1)

    #return train_x, train_y
    #if __name__ == '__main__':
    #train_x, train_y = read_train();
    #train_x.to_csv('train_features.csv', index=False)
    data = pd.concat([train_x, pd.DataFrame({'y':train_y})], axis=1)
    data.to_csv('train_xy2.csv', index=False)

def add_feature():

def read_train(file='train_xy.csv', test=0.2, transform=None):
    print "Read train data..."
    data = pd.read_csv(file)
    # data.drop('enrollment_id','username','course_id', axis=1, inplace=True)
    #x = data.drop(['casual', 'registered', 'count'], axis=1)
    #x = x.values.copy()
    x = pd.concat([data.get(['event_count', 'access_count', 'problem_count', 'page_close_count', 'video_count',
                             'nagivate_count', 'discussion_count', 'wiki_count']),
                   #pd.get_dummies(data.username, prefix='user'),
                   pd.get_dummies(data.course_id, prefix='course')],
                  axis=1)
    scaler = StandardScaler()
    if transform == 'log':
        print "log transform the input features"
        x = scaler.fit_transform(np.log(x + 1.))
    y = data['y']
    x = x.values.astype(np.float32)
    y = y.values.astype(np.int32)
    return x, y


def read_test(file='test_features.csv', transform=None, scaler=None):
    print "Read test data..."
    data = pd.read_csv(file)
    x = pd.concat([data.get(['event_count', 'access_count', 'problem_count', 'page_close_count', 'video_count',
                             'nagivate_count', 'discussion_count', 'wiki_count']),
                   # pd.get_dummies(data.username, prefix='user'),
                   pd.get_dummies(data.course_id, prefix='course')],
                  axis=1)
    enrollment_id = data.values[:,0]
    if transform == 'log':
        print "log transform the input features"
        if scaler == None:
            print("Using different scaler for testing set...")
            scaler = StandardScaler()
        x = scaler.fit_transform(np.log(x + 1.))
    x = x.values.astype(np.float32)
    return x, enrollment_id


def do_RF(train_x, train_y, test_x=None, test_y=None, n_estimators=2000, max_depth=20, max_features=20,
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
        if test_x == None or test_y == None:
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


def do_gbdt(train_x, train_y, test_x=None, test_y=None, learning_rate=0.03, max_depth=8, max_features=25,
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


def do_nn(xTrain, yTrain, test_x=None, test_y=None, dropout_in=0.2, dense0_num=600, dropout_p=0.4, dense1_num=1200,
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
        if test_x != None and test_y != None:
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

def make_predictions(clfs, predict_x, enrollment_id, test_x=None, test_y=None, outfile='test_sub.csv', weights=[]):
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

    predict_y=None
    for clf, wt in zip(clfs, res['x']):
        pred_test_current_clf = pd.DataFrame((clf.predict_proba(predict_x))) * wt
        if predict_y==None:
            predict_y = pred_test_current_clf
        else:
            predict_y = predict_y + pred_test_current_clf

    sub = pd.concat([pd.DataFrame(enrollment_id), pd.DataFrame(predict_y)],axis=1)
    sub.to_csv(outfile, index=False, header=False)


def run():
    train_ratio = 0.9
    test_ratio = 1 - train_ratio
    x, y = read_train(test=0.1)
    sss = StratifiedShuffleSplit(y, test_size=test_ratio, random_state=1234)
    for train_index, test_index in sss:
        break

    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]

    predict_x, enrollment_id = read_test()

    clf_nn = do_nn(train_x, train_y, test_x=test_x, test_y=test_y, dropout_in=0.1, dense0_num=200, dropout_p=0.5,
                   dense1_num=400, update_learning_rate=0.00003, update_momentum=0.9, test_ratio=0.1, max_epochs=20)
    clf_rf = do_RF(train_x, train_y, test_x=test_x, test_y=test_y)
    clf_gbdt = do_gbdt(train_x, train_y, test_x=test_x, test_y=test_y)

    make_predictions([clf_rf, clf_gbdt,clf_nn], predict_x, enrollment_id, test_x=test_x, test_y=test_y,
                     outfile='test_sub1.csv')

def tune():
    train_ratio = 0.9
    test_ratio = 1 - train_ratio
    x, y = read_train(test=0.1)
    sss = StratifiedShuffleSplit(y, test_size=test_ratio, random_state=1234)
    for train_index, test_index in sss:
        break

    train_x, train_y = x[train_index], y[train_index]
    test_x, test_y = x[test_index], y[test_index]

    #do_nn(train_x, train_y, test_x=test_x, test_y=test_y, search=True)
    do_RF(train_x, train_y, test_x=test_x, test_y=test_y, search=True)
    do_gbdt(train_x, train_y, test_x=test_x, test_y=test_y, search=True)



if __name__ == '__main__':
    #make_train()
    #tune()
    run()
