# -*-coding:utf-8-*-
"""
recurrent convolution neural network
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
import numpy as np
from sklearn.model_selection import KFold
import os
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Input
from keras import backend
from keras.layers import Lambda
import pandas as pd
from keras.layers import concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from dl_models.custom import RocCallback
import gc

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


def get_coefs(line):
    lines = line.strip().split()
    return lines[0], np.asarray(lines[1:], dtype='float32')


def create_embedding():
    embedding_index = dict()
    for o in codecs.open(EMBEDDING_FILE, encoding='utf-8'):
        try:
            word, vector = get_coefs(o)
            if len(vector) == 300:
                embedding_index[word] = vector
        except Exception as e:
            print(e)
            continue
    return embedding_index


embedding_index = create_embedding()
all_embs = np.stack(embedding_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = len(word_index) + 1
print('number of word is', nb_words)

# constructing left and right context
sequence_left_context_train = [[len(word_index)] + x[:-1] for x in sequence_train]
sequence_right_context_train = [[len(word_index)] + x[1:] for x in sequence_train]
left_context_train = pad_sequences(sequence_left_context_train, maxlen=MAX_LEN)
right_context_train = pad_sequences(sequence_right_context_train, maxlen=MAX_LEN)
print('left context shape is', left_context_train.shape)
print('right context shape is', right_context_train.shape)

sequence_left_context_test = [[len(word_index)] + x[:-1] for x in sequence_test]
sequence_right_context_test = [[len(word_index)] + x[1:] for x in sequence_test]
left_context_test = pad_sequences(sequence_left_context_test, maxlen=MAX_LEN)
right_context_test = pad_sequences(sequence_right_context_test, maxlen=MAX_LEN)

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_SIZE))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print('embedding shape is', embedding_matrix.shape)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[labels].values
print('labels shape is', y_train.shape)
print('constructing class weight map')
class_weight = dict()
for i in range(len(labels)):
    class_weight[i] = 1 / len(df_train[df_train[labels[i]] == 1])

print('class_weight is', class_weight)
indice_fold = 0


def delete_files(file_folder='./logs'):
    for the_file in os.listdir(file_folder):
        file_path = os.path.join(file_folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


kf = KFold(n_splits=10, shuffle=True, random_state=2)
for idx_train, idx_valid in kf.split(X_train, y_train):
    print('start rnn training {} fold'.format(indice_fold))
    delete_files()
    _X_train = X_train[idx_train]
    _left_context_train = left_context_train[idx_train]
    _right_context_train = right_context_train[idx_train]
    _y_train = y_train[idx_train]
    _X_valid = X_train[idx_valid]
    _left_context_valid = left_context_train[idx_valid]
    _right_context_valid = right_context_train[idx_valid]
    _y_valid = y_train[idx_valid]

    center = Input(shape=(None,), dtype='int32')
    left_context = Input(shape=(None,), dtype='int32')
    right_context = Input(shape=(None,), dtype='int32')
    embedder = Embedding(nb_words, EMBEDDING_SIZE, weights=[embedding_matrix], trainable=True)
    center_embedding = embedder(center)
    left_embedding = embedder(left_context)
    right_embeddig = embedder(right_context)

    forward = LSTM(128, activation='relu', recurrent_dropout=0.2, dropout=0.1, return_sequences=True)(left_embedding)
    backward = LSTM(128, activation='relu', recurrent_dropout=0.2, dropout=0.1, return_sequences=True,
                    go_backwards=True)(right_embeddig)
    concat = concatenate([forward, center_embedding, backward], axis=2)
    semantic = TimeDistributed(Dense(100, activation='tanh'))(concat)
    pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(100,))(semantic)
    output = Dense(6, activation='sigmoid')(pool_rnn)

    model = Model(inputs=[left_context, center, right_context], outputs=output)

    roc_auc_callback = RocCallback([_left_context_train, _X_train, _right_context_train], _y_train,
                                   [_left_context_valid, _X_valid, _right_context_valid], _y_valid)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_save_path = './models/text_cnn_non_static_' + str(indice_fold) + '.h5'
    model_check_point = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
    tb_callback = TensorBoard('./logs', write_graph=True, write_images=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit([_left_context_train, _X_train, _right_context_train],
                     _y_train,
                     batch_size=BATCH_SIZE,
                     epochs=num_epoch,
                     validation_data=([_left_context_valid, _X_valid, _right_context_valid], _y_valid),
                     class_weight=class_weight,
                     shuffle=True,
                     callbacks=[roc_auc_callback, early_stopping, model_check_point, tb_callback])

    print(indice_fold, "validation loss:", min(hist.history["val_loss"]))

    # model_list.append(model)

    submission = pd.DataFrame(
        data=model.predict([left_context_test, X_test, right_context_test], batch_size=BATCH_SIZE,
                           verbose=1),
        columns=labels)
    submission.to_csv('./temp_submissions/temp_' + str(indice_fold) + '.csv', encoding='utf-8', index=False)
    backend.clear_session()

    del model, hist
    gc.collect()
    gc.collect()

    indice_fold += 1

print('start predicting...')
submission = pd.DataFrame(data=np.zeros((len(df_test), len(labels))), columns=labels)
for i in range(10):
    temp = pd.read_csv('./temp_submissions/temp_' + str(i) + '.csv', encoding='utf-8')
    print(temp.shape)
    submission += temp

submission /= 10
submission['id'] = df_test['id']

submission.to_csv('../../submission/han_non_static_submit.csv', encoding='utf-8', index=False)
