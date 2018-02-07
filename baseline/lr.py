# -*- coding:utf-8 -*-
"""
Logistic Regression baseline
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from feature_engineering.feature_extract import train_tfidf_unigram_features
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack

""" 
    features component:
    1. tfidf features
    2. handcraft features
    3. topic features 
"""
features = ['word_count', 'unique_count', 'sentiment', 'subjectivity', 'sentence_count', 'punctuation_count',
            'sentence_length', 'word_avg_length', 'uppercase_count', 'stop_word_count', 'noun_count', 'verb_count',
            'adjective_count', 'topic 0', 'topic 1']


def train_features_merge(*X):
    return hstack((X))


def train_cv(df_train, label):
    """
    gird search to tune highparameter
    :param df_train:
    :param label: predict class
    :return: lr model with special parameters
    """
    # label_weight = df_train.shape[0]/df_train[df_train[label]==1].shape[0]
    lr = LogisticRegression(class_weight='balanced', solver='sag')
    # lr.set_params(class_weight={label:label_weight})
    # lr.set_params(class_weight='balanced')

    '''feature composition'''
    df_handcraft_train = pd.read_csv('../input/train_features.csv', encoding='utf-8')[features].as_matrix()
    df_tfidf_train = train_tfidf_unigram_features()
    print(df_handcraft_train.shape)
    print(df_tfidf_train.shape)
    X_train = train_features_merge(df_handcraft_train, df_tfidf_train)
    # X_train = np.concatenate((df_handcraft_train,df_tfidf_train),axis=1)
    # df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    y_train = df_train[label]

    params = {
        'penalty': ('l1', 'l2'),
        'C': np.arange(0.6, 1.5, 0.1),
        # 'solver': ('liblinear', 'sag', 'saga', 'newton-cg', 'lbfgs'),
        'solver':('liblinear','saga'),
        'max_iter': range(500, 1000, 50)
    }
    clf = GridSearchCV(estimator=lr, param_grid=params, scoring='roc_auc', cv=5)
    clf.fit(X_train, y_train)
    return clf


# def predict


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

if __name__ == '__main__':
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    grid_search = train_cv(df_train, 'toxic')
    print(grid_search.best_params_)
