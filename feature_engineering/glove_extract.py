# -*- coding:utf-8 -*-
"""
extract glove word vector features
"""
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import codecs

EMBEDDING_FILE = '../input/glove.840B.300d.txt'
EMBEDDING_SIZE = 300
MAX_FEATURES = 30000  # number of unique words the rows of embedding matrix
MAX_LEN = 100  # max number of words in a comment to use


# tokenizer = Tokenizer(num_words=MAX_FEATURES)
# tokenizer.fit_on_texts(pd.concat((df_train, df_test))['comment_text'].values)
#
# sequence_train = tokenizer.texts_to_sequences(df_train['comment_text'])
# sequence_test = tokenizer.texts_to_sequences(df_test['comment_text'])


# X_train = pad_sequences(sequence_train, maxlen=MAX_LEN)
# X_test = pad_sequences(sequence_test, maxlen=MAX_LEN)


# def get_coefs(word, *arr):
#     return word, np.asarray(arr, dtype='float32')


def get_coefs(line):
    lines = line.strip().split()
    return lines[0], np.asarray(lines[1:], dtype='float32')


# embedding_index = dict(get_coefs(o.strip().split() for o in codecs.open(EMBEDDING_FILE, encoding='utf-8')))
embedding_index = dict()
for o in codecs.open(EMBEDDING_FILE, encoding='utf-8'):
    try:
        # word, vector = get_coefs(*o.strip().split())
        word, vector = get_coefs(o)
        # vector = np.asarray(vector,dtype='float')
        # print(word)
        # print(vector)
        if len(vector) == 300:
            # print(vector)
            embedding_index[word] = vector
    except:
        continue

all_embedding = np.stack(embedding_index.values())
embedding_mean, embedding_std = all_embedding.mean(), all_embedding.std()


# print(embedding_index)


def get_embedding(sequence, embedding_dict):
    """

    :param sequence:
    :param embedding_dict:
    :return: embedding list
    """
    _len = len(sequence)
    if len(sequence) > MAX_LEN:
        _len = MAX_LEN
        diff = 0
    else:
        diff = MAX_LEN - _len
    sequence_embedding = []
    for i in range(0, _len):
        sequence_embedding.extend(
            embedding_dict.get(sequence[i], np.random.normal(embedding_mean, embedding_std, EMBEDDING_SIZE)))
    sequence_embedding.extend([0] * diff * EMBEDDING_SIZE)
    return sequence_embedding


'''
normal distribution sample for those word not in embedding
'''


def get_train_embedding():
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')
    return df_train['comment_text'].apply(lambda x: get_embedding(x.split(), embedding_index)).values


def get_test_embedding():
    df_test = pd.read_csv('../input/test_clean.csv', encoding='utf-8')
    return df_test['comment_text'].apply(lambda x: get_embedding(x.split(), embedding_index))


# print(get_train_embedding())
print(get_train_embedding().shape)

