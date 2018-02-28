# -*- coding:utf-8 -*-
"""
extract glove word vector features
"""
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import codecs
import gc

EMBEDDING_FILE = '../input/glove.840B.300d.txt'
EMBEDDING_SIZE = 300
MAX_FEATURES = 30000  # number of unique words the rows of embedding matrix
MAX_LEN = 30  # max number of words in a comment to use


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
def create_embedding():
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
    return embedding_index


# print(embedding_index)


def get_embedding(sequence, embedding_dict, default_embedding):
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
            embedding_dict.get(sequence[i], default_embedding))
    sequence_embedding.extend([0] * diff * EMBEDDING_SIZE)
    print(len(sequence_embedding))
    return sequence_embedding


def padding_sequences(sequence, max_len=MAX_LEN):
    """
    padding 0 the sequence into fix length
    :param sequence:
    :param max_len:
    :return:
    """
    if len(sequence) < EMBEDDING_SIZE * max_len:
        diff = EMBEDDING_SIZE * max_len - len(sequence)
        sequence.extend([0] * diff)
    return sequence


'''
normal distribution sample for those word not in embedding
'''


def get_train_embedding():
    embedding_index = create_embedding()
    all_embedding = np.stack(embedding_index.values())
    embedding_mean, embedding_std = all_embedding.mean(), all_embedding.std()
    del all_embedding
    gc.collect()
    default_embedding = np.random.normal(embedding_mean, embedding_std, EMBEDDING_SIZE)
    df_train = pd.read_csv('../input/train_clean.csv', encoding='utf-8')
    train_embedding = []
    for index, item in df_train.iterrows():
        print(index)
        train_embedding.append(get_embedding(item['comment_text'].split(), embedding_index, default_embedding))
    # return df_train['comment_text'].apply(lambda x: get_embedding(x.split(), embedding_index,default_embedding)).values
    return np.asarray(train_embedding, dtype='float16')


def get_test_embedding():
    df_test = pd.read_csv('../input/test_clean.csv', encoding='utf-8')
    embedding_index = create_embedding()
    all_embedding = np.stack(embedding_index.values())
    embedding_mean, embedding_std = all_embedding.mean(), all_embedding.std()
    del all_embedding
    gc.collect()
    default_embedding = np.random.normal(embedding_mean, embedding_std, EMBEDDING_SIZE)
    return df_test['comment_text'].apply(lambda x: get_embedding(x.split(), embedding_index, default_embedding))


# print(get_train_embedding())
# embedding = get_train_embedding()
# print(embedding.shape)
# print(embedding.reshape(len(embedding), MAX_LEN * EMBEDDING_SIZE).shape)
if __name__ == '__main__':
    train_embedding = get_train_embedding()
    train_embedding = pd.DataFrame(train_embedding)
    print('storing train embedding csv...')
    train_embedding.to_csv('../feature_engineering/word_embedding/train_embedding.csv', index=False, encoding='utf-8')
