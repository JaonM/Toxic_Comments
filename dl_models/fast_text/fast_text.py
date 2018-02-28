# -*- coding:utf-8 -*-
"""
fast text model
https://arxiv.org/abs/1607.01759
"""
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dense
from sklearn.metrics import roc_auc_score


def create_ngrams_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngrams(sequences, token_indice, ngram_range=2):
    """
    add ngram features to sequence(input list)

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    :param sequences:
    :param token_indice:
    :param ngram_range:
    :return:
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


# Set Parameters
ngram_range = 2
max_features = 30000
max_len = 500
batch_size = 128
embedding_dim = 100
num_epoch = 100

print('Loading data...')
df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')
df_test = pd.read_csv('../input/train_test.csv', encoding='utf-8')

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(pd.concat((df_train, df_test))['comment_text'].values)

X_train = tokenizer.texts_to_sequences(df_train['comment_text'])
X_test = tokenizer.texts_to_sequences(df_test['comment_text'])

if ngram_range > 1:
    print('adding {}-gram features to sequence'.format(ngram_range))
    ngram_set = set()
    for input_list in X_train:
        for i in range(2, ngram_range + 1):
            ngram_set.update(create_ngrams_set(input_list, i))
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[v]: v for v in token_indice}

    max_features = np.max(list(indice_token.keys())) + 1

    X_train = add_ngrams(X_train, token_indice, ngram_range)
    X_test = add_ngrams(X_test, token_indice, ngram_range)

print('padding sequence...')
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
print('X train shape is', X_train.shape)
print('X test shape is', X_test, shape)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[labels]
print('labels shape is', y_train.shape)

num_split = 10
print('Build {} fold cv Model...'.format(num_split))
skf = StratifiedKFold(n_splits=10, shuffle=True)
indice_fold = 0
for idx_train, idx_val in skf.split(X_train, X_train):
    print('training {} fold', indice_fold)
    X_train = X_train[idx_train]
    y_train = y_train[idx_train]
    X_valid = X_train[idx_val]
    y_valid = y_train[idx_val]

    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(units=6, activation='sigmoid'))
