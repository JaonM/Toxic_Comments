# -*- coding:utf-8 -*-

import pandas as pd

input_path = '../input'

df_train = pd.read_csv(input_path + '/train.csv')
df_test = pd.read_csv(input_path + '/test.csv')

print(df_train.head())

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print(len(df_train))
label_sum = 0
for label in labels:
    sub_dtrain = df_train[df_train[label] == 1]
    label_sum += len(sub_dtrain)
    print('label {} '.format(label) + str(len(sub_dtrain)))

print('label 0 ' + str(len(df_train) - label_sum))

df_train['none'] = (df_train[labels].max(axis=1) == 0).astype(int)

print(df_train.head())
