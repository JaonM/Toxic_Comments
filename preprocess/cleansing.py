# -*- coding:utf-8 -*-

import pandas as pd
import re
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import math

# Aphost lookup dict
APPO = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    # "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    # "didn't": "did not",
    "tryin'": "trying"
}
eng_stopwords = set(stopwords.words('english'))


def clean(comment):
    # comment = comment.lower()
    # remove \n
    # try:
    comment = re.sub('\n', '', comment)
    # except TypeError:
    #     print(comment)
    #     return 'unknown'
    # remove username
    comment = re.sub('\[\[.*\]', '', comment)
    # remove ip
    comment = re.sub('\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}', '', comment)

    wordnetlem = WordNetLemmatizer()
    words = text_to_word_sequence(comment)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [word for word in words if word not in eng_stopwords]
    words = [wordnetlem.lemmatize(word, 'v') for word in words]
    clean_comment = ' '.join(words)
    if len(clean_comment) > 0:
        return clean_comment
    else:
        return 'unknown'


# print(clean('[maqinag]sadsadA\nasdREE'))

if __name__ == '__main__':
    # df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')
    # labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    #
    # df_train['comment_text'].fillna('unknown', inplace=True)
    # df_test['comment_text'].fillna('unknown', inplace=True)
    #
    # df_clean_train = pd.DataFrame()
    # df_clean_test = pd.DataFrame()
    # df_clean_train['id'] = df_train['id']
    # df_clean_test['id'] = df_test['id']
    # df_clean_train['comment_text'] = df_train['comment_text'].apply(lambda x: clean(x))
    # df_clean_test['comment_text'] = df_test['comment_text'].apply(lambda x: clean(x))
    # df_clean_train[labels] = df_train[labels]
    #
    # df_clean_train.to_csv('../input/train_clean.csv', index=False, encoding='utf-8')
    # df_clean_test.to_csv('../input/test_clean.csv', index=False, encoding='utf-8')

    # df_clean_test = pd.read_csv('../input/clean_test.csv')
    #

    # for index, item in df_test.iterrows():
    #     print(item['comment_text'])
    #     item['comment_text'] = clean(item['comment_text'])
    #     print(item['comment_text'])
    # df_test.to_csv('../input/test_clean.csv', index=False, encoding='utf-8')

    df_train_clean = pd.read_csv('../input/train_clean.csv', encoding='utf-8')
    for index, item in df_train_clean.iterrows():
        try:
            comment = re.sub('\n', '', item['comment_text'])
        except TypeError:
            print(str(item['id']) + ' ' + str(item['comment_text']))
