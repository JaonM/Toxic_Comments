# -*- coding:utf-8 -*-
"""
word2vec training model
"""
from gensim.models import Word2Vec
import pandas as pd

df_train = pd.read_csv('../input/train_clean.csv',encoding='utf-8')
df_test = pd.read_csv('../input/test_clean.csv',encoding='utf-8')

def train_corpus(corpus):
    w2c = Word2Vec()