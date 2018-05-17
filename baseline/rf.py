# -*- coding:utf-8 -*-
"""
random forest baseline
tree based method against data set imbalance
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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
from feature_engineering.feature_extract import test_tfidf_char_features
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

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def features_merge(*X):
    return hstack((X))


def resample(X, y):
    """
    resample data set to solve imbalance problem

    :param X:
    :param y:
    :return:
    """
    return smote_tomek_oversampling(X, y)


def train_cv(label):
    """
    cross validation to fine tune hypeparameters
    :param label:
    :return:
    """
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    clf = RandomForestClassifier(n_estimators=1000,
                                 class_weight='balanced',
                                 verbose=1)

    # feature composition
    df_handcraft_train = pd.read_csv('../input/train_features.csv')[features]
    tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    tfidf_char_train = train_tfidf_char_features()
    X_train = hstack(df_handcraft_train, tfidf_unigram_train, tfidf_bigram_train,tfidf_char_train)
    y_train = df_train[label]

    grid_params = {
        'n_estimators': range(1000, 4000, 100),
        'min_samples_leaf': range(2, 10, 1)
    }

    grid_clf = GridSearchCV(estimator=clf, param_grid=grid_params, verbose=1, scoring='roc_auc', cv=5)
    grid_clf.fit(X_train, y_train)
    return grid_clf


def train(label):
    """
    train model process
    :param label: target label list
    :return:
    """
    clf = RandomForestClassifier(n_estimators=1000,
                                 class_weight='balanced',
                                 verbose=1)
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')

    '''feature composition'''
    df_handcraft_train = pd.read_csv('../input/train_features.csv', encoding='utf-8')[features].as_matrix()
    # tfidf_unigram_train = train_tfidf_unigram_features()
    tfidf_bigram_train = train_tfidf_bigram_features()
    # tfidf_char_train = train_tfidf_char_features()
    X_train = features_merge(df_handcraft_train,  tfidf_bigram_train)
    y_train = df_train[label]

    # '''resample the data set'''
    # X_train_resampled, y_train_resampled = resample(X_train, y_train)

    # '''feature selection'''
    # model = SelectFromModel(estimator=clf)
    # X_train_resampled = model.transform(X_train_resampled)

    # '''train test split'''
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
    #                                                       random_state=2)
    # clf.fit(X_train, y_train)
    # y_predict = clf.predict(X_valid)
    # print(label + ' roc auc score is ' + str(roc_auc_score(y_valid, y_predict)))

    clf.fit(X_train, y_train)
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
    # tfidf_unigram_test = test_tfidf_unigram_features()
    tfidf_bigram_test = test_tfidf_bigram_features()
    # tfidf_char_test = test_tfidf_char_features()
    X_test = features_merge(df_handcraft_test, tfidf_bigram_test)

    '''predict label'''
    target = clf.predict(X_test)
    df_predict[label] = target
    return df_predict


if __name__ == '__main__':
    # df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    # grid_search = train_cv('toxic')
    # print(grid_search.best_params_)

    df_test = pd.read_csv('../input/test.csv', encoding='utf-8')
    df_predict = pd.DataFrame()
    df_predict['id'] = df_test['id']
    for label in labels:
        clf = train(label)
        df_predict = predict(df_predict, clf, label)
    df_predict.to_csv('../submission/rf_submission.csv', encoding='utf-8', index=False)
