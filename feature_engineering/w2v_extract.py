# -*- coding:utf-8 -*-

from gensim.models import Word2Vec
import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
import gc

EMBEDDING_SIZE = 50
MAX_FEATURES = 30000  # number of unique words the rows of embedding matrix
MAX_LEN = 40

df_train = pd.read_csv('../../input/train_clean.csv', encoding='utf-8')
df_test = pd.read_csv('../../input/test_clean.csv', encoding='utf-8')


def train_corpus():
    corpus = pd.concat((df_train, df_test), axis=0)
    sentences = corpus['comment_text'].apply(lambda x: x.split())
    w2c = Word2Vec(sentences=sentences,
                   iter=100,
                   min_count=5,
                   size=EMBEDDING_SIZE)
    w2c.save('../input/toxic_vectors-negative50')


def create_embedding():
    embedding_index = dict()
    tokenizer = Tokenizer()
    w2v = Word2Vec.load('../../input/toxic_vectors-negative100')
    tokenizer.fit_on_texts(pd.concat((df_train, df_test), axis=0)['comment_text'].values)
    for k in tokenizer.word_index.keys():
        # print(k)
        try:
            embedding_index[k] = w2v[k]
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
        # diff = 0
    # else:
    # diff = MAX_LEN - _len
    sequence_embedding = []
    for i in range(0, _len):
        sequence_embedding.extend(
            embedding_dict.get(sequence[i], default_embedding))
    # sequence_embedding.extend([0] * diff * EMBEDDING_SIZE)
    sequence_embedding = padding_sequences(sequence_embedding)
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
    # train_embedding = get_train_embedding()
    # train_embedding = pd.DataFrame(train_embedding)
    # print('storing train embedding csv...')
    # train_embedding.to_csv('../feature_engineering/word_embedding/w2v_train_embedding.csv', index=False,
    #                        encoding='utf-8')
    train_corpus()
