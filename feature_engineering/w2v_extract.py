# -*- coding:utf-8 -*-
"""
word2vec training model
"""
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')
df_test = pd.read_csv('../input/test_clean.csv', encoding='utf-8')


def train_corpus(corpus):
    w2c = Word2Vec(sentences=corpus, iter=100, size=100, min_count=5)
    w2c.save('../input/toxic_vectors-negative100')


if __name__ == '__main__':
    df = pd.concat((df_train, df_test), axis=0)['comment_text'].apply(lambda x: x.split())
    print(df.values)
    train_corpus(np.asarray(df.values))
    # model = KeyedVectors.load_word2vec_format('../input/toxic_vectors-negative100.bin', binary=True,
    #                                           unicode_errors='ignore')
    model = Word2Vec.load('../input/toxic_vectors-negative100')
    print(model.most_similar('fuck'))
