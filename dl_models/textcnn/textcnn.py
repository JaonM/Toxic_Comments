# -*- coding:utf-8 -*-
"""
text cnn
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

EMBEDDING_FILE = '../../input/glove.840B.300d.txt'
EMBEDDING_SIZE = 300
MAX_FEATURES = 30000  # number of unique words the rows of embedding matrix
MAX_LEN = 300  # max number of words in a comment to use
BATCH_SIZE = 128
num_epoch = 100

df_train = pd.read_csv('../../input/train_clean.csv', encoding='utf-8')
df_test = pd.read_csv('../../input/test_clean.csv', encoding='utf-8')

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(pd.concat((df_train, df_test))['comment_text'].values)

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


embedding_index = create_embedding()
all_embs = np.stack(embedding_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(MAX_FEATURES, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_SIZE))
for word, i in word_index.items():
    if i >= MAX_FEATURES:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[labels].values
print('labels shape is', y_train.shape)

nb_words = len(word_index) + 1

kf = KFold(n_splits=10, shuffle=True, random_state=2)

indice_fold = 0
model_list = []
for idx_train, idx_valid in kf.split(X=X_train, y=y_train):
    print('start training {} fold'.format(indice_fold))
    _X_train = X_train[idx_train]
    _y_train = y_train[idx_train]
    _X_valid = X_train[idx_valid]
    _y_valid = y_train[idx_valid]

    _input = Input(shape=(MAX_LEN,))
    embedding = Embedding(nb_words, EMBEDDING_SIZE, input_length=MAX_LEN)
    embedding_input = embedding(_input)

    # cnn1 模块 kernal size=3
    conv1_1 = Convolution1D(256, kernel_size=3, padding='same')(embedding_input)
    bn1_1 = BatchNormalization()(conv1_1)
    act1_1 = Activation(activation='relu')(bn1_1)
    covn1_2 = Convolution1D(128, kernel_size=3, padding='same')(act1_1)
    bn1_2 = BatchNormalization()(cnn1_2)
    act1_2 = Activation('relu')(bn1_2)
    cnn1 = MaxPooling1D(pool_size=4)(act1_2)

    # cnn2 模块 kernal size=4
    conv2_1 = Convolution1D(256, kernel_size=4, padding='same')(embedding_input)
    bn2_1 = BatchNormalization()(conv2_1)
    act2_1 = Activation(activation='relu')(bn2_1)
    covn2_2 = Convolution1D(128, kernel_size=3, padding='same')(act2_1)
    bn2_2 = BatchNormalization()(covn2_2)
    act2_2 = Activation('relu')(bn2_2)
    cnn2 = MaxPooling1D(pool_size=4)(act2_2)

    # cnn3 模块 kernal size=5
    conv3_1 = Convolution1D(256, kernel_size=5, padding='same')(embedding_input)
    bn3_1 = BatchNormalization()(conv3_1)
    act3_1 = Activation(activation='relu')(bn3_1)
    covn3_2 = Convolution1D(128, kernel_size=3, padding='same')(act3_1)
    bn3_2 = BatchNormalization()(covn3_2)
    act3_2 = Activation('relu')(bn3_2)
    cnn3 = MaxPooling1D(pool_size=4)(act3_2)

    # concatenate
    merge = concatenate([cnn1, cnn2, cnn3])
    merge = Flatten()(merge)
    merge = Dropout(0.5)(merge)
    merge = Dense(512)(merge)   # linear layer
    merge = BatchNormalization()(merge)

    out = Dense(6, activation='sigmoid')(merge)
    model = Model(inputs=_input, outputs=out)

    roc_auc_callback = RocCallback(_X_train, _y_train, _X_valid, _y_valid)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_save_path = './text_cnn_' + str(indice_fold) + '.h5'
    model_check_point = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(_X_train,
                     _y_train,
                     batch_size=BATCH_SIZE,
                     epochs=num_epoch,
                     validation_data=(_X_valid, _y_valid),
                     # class_weight=class_weight,
                     shuffle=True,
                     callbacks=[roc_auc_callback, early_stopping, model_check_point])

    print(indice_fold, "validation loss:", min(hist.history["val_loss"]))

    model_list.append(model)

    indice_fold += 1

print('start predicting...')
submission = pd.DataFrame(data=np.zeros((len(df_test), len(labels))), columns=labels)
for model in model_list:
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    print(preds.shape)
    preds = pd.DataFrame(data=preds, columns=labels)
    print(preds)
    submission += preds

submission /= len(model_list)
submission['id'] = df_test['id']

submission.to_csv('../../submission/fast_text_submit.csv', encoding='utf-8', index=False)
