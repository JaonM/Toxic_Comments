# -*- coding:utf-8 -*-
"""
convolution+lstm
https://arxiv.org/pdf/1511.08630.pdf
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import codecs
from sklearn.model_selection import KFold
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dl_models.custom import RocCallback
from keras.callbacks import TensorBoard
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import AveragePooling1D
from keras.layers import GaussianNoise
from keras import backend as K
import gc
import os

# from feature_engineering.w2v_extract import create_embedding

EMBEDDING_FILE = '../../input/glove.840B.300d.txt'
EMBEDDING_SIZE = 300
MAX_FEATURES = 30000  # number of unique words the rows of embedding matrix
MAX_LEN = 100  # max number of words in a comment to use
BATCH_SIZE = 128
num_epoch = 100

df_train = pd.read_csv('../../input/train_clean.csv', encoding='utf-8')
df_test = pd.read_csv('../../input/test_clean.csv', encoding='utf-8')

tokenizer = Tokenizer(num_words=MAX_FEATURES)
# tokenizer.fit_on_texts(pd.concat((df_train, df_test))['comment_text'].values)
tokenizer.fit_on_texts(df_train['comment_text'].values)

sequence_train = tokenizer.texts_to_sequences(df_train['comment_text'])
sequence_test = tokenizer.texts_to_sequences(df_test['comment_text'])

X_train = pad_sequences(sequence_train, maxlen=MAX_LEN)
X_test = pad_sequences(sequence_test, maxlen=MAX_LEN)

print('train data shape is', X_train.shape)
print('test data shape is', X_test.shape)


# def get_coefs(word, *arr):
#     return word, np.asarray(arr, dtype='float32')


def get_coefs(line):
    lines = line.strip().split()
    return lines[0], np.asarray(lines[1:], dtype='float32')


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


embedding_index = create_embedding()
all_embs = np.stack(embedding_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = len(word_index) + 1
print('number of word is', nb_words)

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_SIZE))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('embedding shape is', embedding_matrix.shape)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[labels].values
print('labels shape is', y_train.shape)

kf = KFold(n_splits=10, shuffle=True, random_state=2)

print('constructing class weight map')
class_weight = dict()
for i in range(len(labels)):
    class_weight[i] = 1 / len(df_train[df_train[labels[i]] == 1])

print('class_weight is', class_weight)
indice_fold = 0


# model_list = []

def delete_files(file_folder='./logs'):
    for the_file in os.listdir(file_folder):
        file_path = os.path.join(file_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


# statics hand-craft features
# statics_train = pd.read_csv('../feature_engineering/statics_train.csv',encoding='utf-8').as_matrix()
# statics_test = pd.read_csv('../feature_engineering/statics_test.csv',encoding='utf-8').as_matrix()

for idx_train, idx_valid in kf.split(X=X_train, y=y_train):
    print('start training {} fold'.format(indice_fold))
    delete_files()
    _X_train = X_train[idx_train]
    _y_train = y_train[idx_train]
    _X_valid = X_train[idx_valid]
    _y_valid = y_train[idx_valid]

    # _statics_train = statics_train[idx_train]
    # _statics_valid = statics_train[idx_valid]

    # _statics_input = Input(shape=(statics_train.shape[1],))

    _input = Input(shape=(MAX_LEN,))
    embedding = Embedding(nb_words, EMBEDDING_SIZE, input_length=MAX_LEN, weights=[embedding_matrix], trainable=True)
    embedding_input = embedding(_input)

    # cnn1 模块 kernal size=1
    conv1 = Convolution1D(128, kernel_size=1, padding='causal', activation='relu')(embedding_input)

    # cnn2 模块 kernal size=2
    conv2 = Convolution1D(128, kernel_size=2, padding='causal', activation='relu')(embedding_input)

    # cnn3 模块 kernal size=3
    conv3 = Convolution1D(128, kernel_size=3, padding='causal', activation='relu')(embedding_input)

    # concatenate
    cnns = concatenate([conv1, conv2, conv3])

    lstm = Bidirectional(LSTM(256, activation='relu', recurrent_dropout=0.1, dropout=0.2))(cnns)
    # dense = BatchNormalization()(lstm)
    dense = Dense(128, activation='relu')(lstm)
    dense = Dropout(0.4)(dense)
    dense = BatchNormalization()(dense)
    out = Dense(6, activation='sigmoid')(dense)

    model = Model(inputs=_input, outputs=out)

    roc_auc_callback = RocCallback(_X_train, _y_train, _X_valid, _y_valid)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_save_path = './models/clstm_non_static_' + str(indice_fold) + '.h5'
    model_check_point = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
    tb_callback = TensorBoard('./logs', write_graph=True, write_images=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(_X_train,
                     _y_train,
                     batch_size=BATCH_SIZE,
                     epochs=num_epoch,
                     validation_data=(_X_valid, _y_valid),
                     class_weight=class_weight,
                     shuffle=True,
                     callbacks=[roc_auc_callback, early_stopping, model_check_point, tb_callback])

    print(indice_fold, "validation loss:", min(hist.history["val_loss"]))

    # model_list.append(model)

    submission = pd.DataFrame(data=model.predict(X_test, batch_size=BATCH_SIZE, verbose=1), columns=labels)
    submission.to_csv('./temp_submissions/clstm_temp_' + str(indice_fold) + '.csv', encoding='utf-8', index=False)
    K.clear_session()

    del model, hist
    gc.collect()
    gc.collect()

    indice_fold += 1

print('start predicting...')
submission = pd.DataFrame(data=np.zeros((len(df_test), len(labels))), columns=labels)
for i in range(10):
    temp = pd.read_csv('./temp_submissions/clstm_temp_' + str(i) + '.csv', encoding='utf-8')
    print(temp.shape)
    submission += temp

submission /= 10
submission['id'] = df_test['id']

submission.to_csv('../../submission/clstm_non_static_submit.csv', encoding='utf-8', index=False)
