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
from keras.models import Model
from keras.layers import Input
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers.merge import concatenate
from dl_models.custom import RocCallback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def create_ngrams_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngrams(sequences, token_indice, ngram_range=2):
    """
    add ngram features to sequence(input list)

    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
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
max_len = 400
batch_size = 64
embedding_dim = 100
num_epoch = 50

print('Loading data...')
df_train = pd.read_csv('../../input/train_clean.csv', encoding='utf-8')
df_test = pd.read_csv('../../input/test_clean.csv', encoding='utf-8')

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
print('X test shape is', X_test.shape)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[labels].values
print('labels shape is', y_train.shape)

print('constucting class weight map')
class_weight = dict()
for i in range(len(labels)):
    class_weight[i] = len(df_train) / len(df_train[df_train[labels[i]] == 1])

num_split = 10
print('Build {} fold cv Model...'.format(num_split))
# skf = StratifiedKFold(n_splits=10, shuffle=True)
kf = KFold(n_splits=num_split, shuffle=True, random_state=2)
indice_fold = 0

model_list = list()
for idx_train, idx_val in kf.split(X=X_train, y=y_train):
    print('training {} fold'.format(indice_fold))
    _X_train = X_train[idx_train]
    _y_train = y_train[idx_train]
    _X_valid = X_train[idx_val]
    _y_valid = y_train[idx_val]

    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=max_len, trainable=True))
    model.add(GlobalAveragePooling1D())
    # model.add(GlobalMaxPooling1D())
    model.add(Dense(units=6, activation='sigmoid'))

    embedding = Embedding(max_features, embedding_dim, input_length=max_len, trainable=True)
    input_avg = Input(shape=(max_len,), dtype='int32')
    input_max = Input(shape=(max_len,), dtype='int32')

    avg_embedding = embedding(input_avg)
    max_embedding = embedding(input_avg)

    global_avg = GlobalAveragePooling1D()(avg_embedding)
    global_max = GlobalMaxPooling1D()(max_embedding)

    merge = concatenate([global_avg, global_max])

    output = Dense(6,activation='sigmoid')(merge)

    model = Model(inputs=[input_avg,input_max],outputs=output)

    roc_auc_callback = RocCallback(_X_train, _y_train, _X_valid, _y_valid)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_save_path = './fast_text_' + str(indice_fold) + '.h5'
    model_check_point = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(_X_train,
              _y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              validation_data=(_X_valid, _y_valid),
              class_weight=class_weight,
              callbacks=[roc_auc_callback, early_stopping, model_check_point])

    model_list.append(model)

    indice_fold += 1

print('start predicting...')
submission = pd.DataFrame(data=np.zeros((len(df_test), len(labels))), columns=labels)
for model in model_list:
    preds = model.predict(X_test, batch_size=batch_size, verbose=1)
    preds = pd.DataFrame(data=preds.ravel(), columns=labels)
    print(preds)
    submission += preds

submission /= len(labels)
submission['id'] = df_test['id']

submission.to_csv('../submission/fast_text_submit.csv', encoding='utf-8', index=False)
