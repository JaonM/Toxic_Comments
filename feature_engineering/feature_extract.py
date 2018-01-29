# -*- coding:utf-8 -*-
"""
basic features + tfidf feature
"""

import pandas as pd
import numpy as np
import math
import re
from preprocess.cleansing import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
from nltk.corpus import stopwords


def tfidf_corpus():
    df_corpus = pd.concat((df_train, df_test), axis=0)
    df_corpus['comment_text'].apply(lambda x: clean(x))
    tfidfVec = TfidfVectorizer(
        strip_accents='unicode',
        max_features=100000,
        ngram_range=(1, 1),
        analyzer='word',
    )
    return tfidfVec.fit_transform(df_corpus['comment_text'])


# print(tfidf_corpus())

def word_count(comment):
    return len(comment.split())


def unique_word_count(comment):
    return len(set(comment.split()))


def sentiment(comment):
    """

    :param comment:
    :return: sentiment polarity -1~1
    """
    sentence = TextBlob(comment)
    return sentence.sentiment.polarity


def subjectivity(comment):
    """

    :param comment:
    :return: score 0-1
    """
    return TextBlob(comment).sentiment.subjectivity


def sentence_count(comment):
    """
    use raw sentence
    :param comment:
    :return:
    """
    return len(TextBlob(comment).sentences)


def punctuation_count(comment):
    """
    use raw sentence
    :param comment:
    :return:
    """
    return len(
        re.findall(r'\||"|!|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|\||}|~|\t|\n', comment))


def sentence_length(comment):
    """
    use raw sentence
    :param comment:
    :return:
    """
    return len(comment)


def word_avg_length(comment):
    _sum = 0
    for word in comment.split():
        _sum += len(word)
    if word_count(comment) == 0:
        return 0
    else:
        return _sum / word_count(comment)


def uppercase_count(comment):
    count = 0
    for word in comment.split():
        if word.isupper():
            count += 1
    return count


def stop_word_count(comment):
    """
    use raw sentence
    :param comment:
    :return:
    """
    count = 0
    for word in comment.split():
        if word in set(stopwords.words('english')):
            count += 1
    return count


def noun_count(comment):
    sentence = TextBlob(comment)
    count = 0
    for tag in sentence.tags:
        if re.search('N', tag[1]):
            count += 1
    return count


def verb_count(comment):
    sentence = TextBlob(comment)
    count = 0
    for tag in sentence.tags:
        if re.search('V', tag[1]):
            count += 1
    return count


def adjective_count(comment):
    sentence = TextBlob(comment)
    count = 0
    for tag in sentence.tags:
        if re.search('JJ', tag[1]):
            count += 1
    return count


# print(adjective_count('i love this world which was beloved by all the people here'))

if __name__ == '__main__':
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    df_train_clean = pd.read_csv('../input/train_clean.csv')
    df_test_clean = pd.read_csv('../input/test_clean.csv')

    # tfidf_feats = tfidf_corpus()
    train_feats = pd.DataFrame()
    train_feats['id'] = df_train['id']
    train_feats['word_count'] = df_train_clean['comment_text'].apply(lambda x: word_count(x))
    train_feats['unique_count'] = df_train_clean['comment_text'].apply(lambda x: unique_word_count(x))
    train_feats['sentiment'] = df_train_clean['comment_text'].apply(lambda x: sentiment(x))
    train_feats['subjectivity'] = df_train_clean['comment_text'].apply(lambda x: subjectivity(x))
    train_feats['sentence_count'] = df_train['comment_text'].apply(lambda x: sentence_count(x))
    train_feats['punctuation_count'] = df_train['comment_text'].apply(lambda x: punctuation_count(x))
    train_feats['sentence_length'] = df_train['comment_text'].apply(lambda x: sentence_length(x))
    train_feats['word_avg_length'] = df_train_clean['comment_text'].apply(lambda x: word_avg_length(x))
    train_feats['uppercase_count'] = df_train['comment_text'].apply(lambda x: uppercase_count(x))
    train_feats['stop_word_count'] = df_train['comment_text'].apply(lambda x: stop_word_count(x))
    train_feats['noun_count'] = df_train_clean['comment_text'].apply(lambda x: noun_count(x))
    train_feats['verb_count'] = df_train_clean['comment_text'].apply(lambda x: verb_count(x))
    train_feats['adjective_count'] = df_train_clean['comment_text'].apply(lambda x: adjective_count(x))

    train_feats.to_csv('../input/train_features.csv', index=False, encoding='utf-8')

    test_feats = pd.DataFrame()
    test_feats['id'] = df_train['id']
    test_feats['word_count'] = df_test_clean['comment_text'].apply(lambda x: word_count(x))
    test_feats['unique_count'] = df_test_clean['comment_text'].apply(lambda x: unique_word_count(x))
    test_feats['sentiment'] = df_test_clean['comment_text'].apply(lambda x: sentiment(x))
    test_feats['subjectivity'] = df_test_clean['comment_text'].apply(lambda x: subjectivity(x))
    test_feats['sentence_count'] = df_test['comment_text'].apply(lambda x: sentence_count(x))
    test_feats['punctuation_count'] = df_test['comment_text'].apply(lambda x: punctuation_count(x))
    test_feats['sentence_length'] = df_test['comment_text'].apply(lambda x: sentence_length(x))
    test_feats['word_avg_length'] = df_test_clean['comment_text'].apply(lambda x: word_avg_length(x))
    test_feats['uppercase_count'] = df_test['comment_text'].apply(lambda x: uppercase_count(x))
    test_feats['stop_word_count'] = df_test['comment_text'].apply(lambda x: stop_word_count(x))
    test_feats['noun_count'] = df_test_clean['comment_text'].apply(lambda x: noun_count(x))
    test_feats['verb_count'] = df_test_clean['comment_text'].apply(lambda x: verb_count(x))
    test_feats['adjective_count'] = df_test_clean['comment_text'].apply(lambda x: adjective_count(x))
    test_feats.to_csv('../input/test_features.csv', index=False, encoding='utf-8')
