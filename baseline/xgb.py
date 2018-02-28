# -*- coding:utf-8 -*-
"""
xgboost baseline
"""

from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
from feature_engineering.feature_extract import train_tfidf_unigram_features
from feature_engineering.feature_extract import train_tfidf_bigram_features
from feature_engineering.feature_extract import train_tfidf_char_features
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from feature_engineering.feature_extract import test_tfidf_bigram_features
from feature_engineering.feature_extract import test_tfidf_unigram_features
from feature_engineering.feature_extract import test_tfidf_char_features
from sklearn.feature_selection import SelectFromModel
from baseline.resample import smote_tomek_oversampling
from sklearn.model_selection import StratifiedKFold
from feature_engineering.glove_extract import get_train_embedding
from feature_engineering.glove_extract import get_test_embedding

""" 
    features component:
    1. tfidf features
    2. handcraft features
    3. topic features 
"""
features = ['word_count', 'unique_count', 'sentiment', 'subjectivity', 'sentence_count', 'punctuation_count',
            'sentence_length', 'word_avg_length', 'uppercase_count', 'stop_word_count', 'noun_count', 'verb_count',
            'adjective_count', 'topic 0', 'topic 1']

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def features_merge(*X):
    return hstack((X))


def train_grid_search(label):
    """
    better params:
    -   colsample_bytree=0.8
    -   colsample_bylevel=0.6
    -   gamma:2
    -   max_depth=4
    cross validation to fine tune hypeparameters
    :param label:
    :return:
    """

    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')

    _toxic_count = len(df_train[df_train.max(axis=1) == 0])
    print(_toxic_count)
    _clean_count = len(df_train) - _toxic_count

    clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=4,
                        silent=False,
                        # scale_pos_weight=_toxic_count / _clean_count,
                        colsample_bytree=0.8,
                        colsample_bylevel=0.6,
                        gamma=2,
                        objective='binary:logistic')

    # feature composition
    df_handcraft_train = pd.read_csv('../feature_engineering/statics/statics_train.csv')
    tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    tfidf_char_train = train_tfidf_char_features()
    # word_embedding_train = pd.read_csv('../feature_engineering/word_embedding/train_embedding.csv', encoding='utf-8')
    # print(word_embedding_train.shape)
    # X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train, tfidf_char_train,
    #                          word_embedding_train)
    X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train, tfidf_char_train)
    # X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_char_train)
    y_train = df_train[label]

    grid_params = {
        # 'learning_rate': np.arange(0.08, 0.2, 0.01),
        # 'n_estimators': range(1000, 4000, 100),
        # 'gamma': range(0, 5, 1),
        # 'max_depth': range(4, 5, 1),
        # 'colsample_bytree': np.arange(0.6, 1, 0.1),
        'colsample_bylevel': np.arange(0.4, 1.0, 0.1)
    }

    # '''resample the data set'''
    # X_train_resampled, y_train_resampled = resample(X_train, y_train)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                          random_state=2)
    print('grid search construct')
    grid_clf = GridSearchCV(estimator=clf, param_grid=grid_params, verbose=1, scoring='roc_auc', cv=5)
    grid_clf.fit(X=X_train, y=y_train, eval_metric='auc', eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
    return grid_clf


def resample(X, y):
    """
    resample data set to solve imbalance problem

    :param X:
    :param y:
    :return:
    """
    return smote_tomek_oversampling(X, y)


def train(label):
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=8,
                        silent=True,
                        objective='binary:logistic')

    '''feature composition'''
    df_handcraft_train = pd.read_csv('../input/train_features.csv', encoding='utf-8')[features].as_matrix()
    tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    tfidf_char_train = train_tfidf_char_features()
    X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train, tfidf_char_train)
    y_train = df_train['label']

    # '''resample the data set'''
    # X_train_resampled, y_train_resampled = resample(X_train, y_train)

    '''feature selection'''
    model = SelectFromModel(estimator=clf)
    X_train = model.transform(X_train)

    '''train test split'''
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                          random_state=2)
    clf.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
    y_predict = clf.predict(X_valid)
    print(label + ' roc auc score is ' + str(roc_auc_score(y_valid, y_predict)))

    # clf.fit(X_train_resampled, y_train_resampled)
    return clf


def train_cv(label):
    """
    cross validation training process
    :param label:
    :return: list clfs
    """
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    num_fold = 5
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True)
    count = 0
    clf_list = []
    for idx_train, idx_valid in skf.split(df_train[label], df_train[label]):
        print("fitting fold " + str(count))
        clf = _train(label, idx_train, idx_valid, count)
        count += 1
        clf_list.append(clf)


def predict_cv(df_predict, label, classifier_list):
    """
    average ensemble predict
    :param df_train:
    :param label:
    :param classifier_list:
    :return:
    """
    '''feature composition'''
    df_handcraft_test = pd.read_csv('../feature_engineering/test_features.csv', encoding='utf-8')
    tfidf_unigram_test = test_tfidf_unigram_features()
    tfidf_bigram_test = test_tfidf_bigram_features()
    tfidf_char_test = test_tfidf_char_features()
    word_embedding_test = get_test_embedding()
    X_test = features_merge(df_handcraft_test, tfidf_unigram_test, tfidf_bigram_test, tfidf_char_test,
                            word_embedding_test)

    '''predict label average'''
    for clf in classifier_list:
        target = clf.predict(X_test)
        df_predict[label] += target
    df_predict[label] = df_predict[label] / len(classifier_list)
    return df_predict


def _train(label, idx_train, idx_valid, index):
    """
    :param label:
    :param idx_train: train index
    :param idx_valid: valid index
    :param index: number of training process
    :return: classifier
    """
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,
                        max_depth=8,
                        silent=True,
                        objective='binary:logistic')

    '''feature composition'''
    df_handcraft_train = pd.read_csv('../input/train_features.csv', encoding='utf-8')[features].as_matrix()
    tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    tfidf_char_train = train_tfidf_char_features()
    X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train, tfidf_char_train)
    y_train = df_train['label']

    # '''resample the data set'''
    # X_train_resampled, y_train_resampled = resample(X_train, y_train)

    '''feature selection'''
    model = SelectFromModel(estimator=clf)
    X_train = model.transform(X_train)

    '''train test split'''
    X_train = X_train[idx_train]
    y_train = y_train[idx_train]
    X_valid = X_train[idx_valid]
    y_valid = X_train[idx_valid]
    clf.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_valid, y_valid)], early_stopping_rounds=20)
    y_predict = clf.predict(X_valid)
    print('cv ' + str(count) + ' ' + label + ' roc auc score is ' + str(roc_auc_score(y_valid, y_predict)))

    # clf.fit(X_train_resampled, y_train_resampled)
    return clf


def predict(df_predict, clf, label):
    """
    model predict process
    :param df_predict:predict dataframe
    :param clf: classifier
    :param label: label
    :return: predicted test dataframe
    """

    '''feature composition'''
    df_handcraft_test = pd.read_csv('../input//test_features.csv', encoding='utf-8')[features]
    tfidf_unigram_test = test_tfidf_unigram_features()
    tfidf_bigram_test = test_tfidf_bigram_features()
    tfidf_char_test = test_tfidf_char_features()
    X_test = features_merge(df_handcraft_test, tfidf_unigram_test, tfidf_bigram_test, tfidf_char_test)

    '''predict label'''
    target = clf.predict(X_test)
    df_predict[label] = target
    return df_predict


if __name__ == '__main__':
    '''predict and submit'''
    # submission = pd.DataFrame()
    # df_test = pd.read_csv('../input/test.csv', encoding='utf-8')
    # submission['id'] = df_test['id']
    # for label in labels:
    #     clf = train(label)
    #     submission = predict_cv(submission, clf, label)
    # submission.to_csv('../submission/xgb_submission.csv', encoding='utf-8', index=False)

    '''fine tune parameters'''
    grid_search = train_grid_search('toxic')
    print(grid_search.best_params_)
