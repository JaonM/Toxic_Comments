# -*- coding:utf-8 -*-
"""
Logistic Regression baseline
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from feature_engineering.feature_extract import train_tfidf_unigram_features
from feature_engineering.feature_extract import train_tfidf_char_features
from feature_engineering.feature_extract import train_tfidf_bigram_features
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from baseline.resample import smote_tomek_oversampling
from sklearn.metrics import roc_auc_score
from feature_engineering.feature_extract import test_tfidf_bigram_features
from feature_engineering.feature_extract import test_tfidf_unigram_features
from sklearn.feature_selection import SelectFromModel

""" 
    features component:
    1. tfidf features
    2. handcraft features
    3. topic features 
"""
features = ['word_count', 'unique_count', 'sentiment', 'subjectivity', 'sentence_count', 'punctuation_count',
            'sentence_length', 'word_avg_length', 'uppercase_count', 'stop_word_count', 'noun_count', 'verb_count',
            'adjective_count', 'topic 0', 'topic 1']


def features_merge(*X):
    return hstack((X))


def train_cv(label):
    """
    gird search to tune hypeparameters
    :param label: target label
    :return: lr model with special parameters
    """
    # label_weight = df_train.shape[0]/df_train[df_train[label]==1].shape[0]
    lr = LogisticRegression(class_weight='balanced', solver='sag', random_state=22, verbose=1, max_iter=6000)
    # lr.set_params(class_weight={label:label_weight})
    # lr.set_params(class_weight='balanced')

    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    '''feature composition'''
    df_handcraft_train = pd.read_csv('../input/train_features.csv', encoding='utf-8')[features].as_matrix()
    tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    # tfidf_char_train = train_tfidf_char_features()
    print(df_handcraft_train.shape)
    # print(tfidf_train.shape)
    X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train)
    # X_train = np.concatenate((df_handcraft_train,df_tfidf_train),axis=1)
    # df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    y_train = df_train[label]

    params = {
        # 'penalty': ('l1', 'l2'),
        'C': np.arange(1, 5, 1),
        # 'solver': ('liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'),
        'solver': ('sag', 'saga', 'newton-cg'),
        'max_iter': range(6000, 7000, 100)

    }

    clf = GridSearchCV(estimator=lr, param_grid=params, scoring='roc_auc', cv=5, verbose=1)
    clf.fit(X_train, y_train)
    return clf


def resample(X, y):
    """
    resample data set to solve imbalance problem

    :param X:
    :param y:
    :return:
    """
    return smote_tomek_oversampling(X, y)


def train(label):
    """
    train model process
    :param label: target label list
    :return:
    """
    clf = LogisticRegression(class_weight='balanced', solver='sag', random_state=22, verbose=1, max_iter=6000)
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')

    '''feature composition'''
    df_handcraft_train = pd.read_csv('../input/train_features.csv', encoding='utf-8')[features].as_matrix()
    tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    X_train = features_merge(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train)
    y_train = df_train['label']

    '''resample the data set'''
    X_train_resampled, y_train_resampled = resample(X_train, y_train)

    '''feature selection'''
    model = SelectFromModel(estimator=clf)
    X_train_resampled = model.transform(X_train_resampled)

    '''train test split'''
    X_train, y_train, X_valid, y_valid = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2,
                                                          random_state=2)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_valid)
    print(label + ' roc auc score is ' + str(roc_auc_score(y_valid, y_predict)))

    clf.fit(X_train_resampled, y_train_resampled)
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
    X_test = features_merge(df_handcraft_test, tfidf_unigram_test, tfidf_bigram_test)

    '''predict label'''
    target = clf.predict(X_test)
    df_predict[label] = target
    return predict


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

if __name__ == '__main__':
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    grid_search = train_cv('toxic')
    print(grid_search.best_params_)

    df_test = pd.read_csv('../input/test.csv', encoding='utf-8')
    df_predict = pd.DataFrame()
    df_predict['id'] = df_test['id']
    for label in labels:
        clf = train(label)
        df_predict = predict(df_predict, clf, label)
    df_predict.to_csv('lr_submission', encoding='utf-8', index=False)
