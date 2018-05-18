# -*- coding:utf-8 -*-

import re

import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob

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
    words = text_to_word_sequence(comment, filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
    words = [APPO[word] if word in APPO else word for word in words]
    words = [word for word in words if word not in eng_stopwords]
    words = [TextBlob(word).correct().string for word in words]
    words = [wordnetlem.lemmatize(word, 'v') for word in words]
    clean_comment = ' '.join(words)
    if len(clean_comment) > 0:
        print(clean_comment)
        return clean_comment
    else:
        return 'unknown'


def correct_translate(comment):
    text = TextBlob(comment)
    # lang = text.detect_language()
    # print(lang)
    # if lang != 'en':
    #     try:
    #         text = text.translate(from_lang=lang, to='en')
    #     except:
    #         pass
    text = text.correct()
    print(text.string)
    return text.string


# print(clean('[maqinag]sadsadA\nasdREE'))

if __name__ == '__main__':
    df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
    df_train['comment_text'].apply(lambda x: clean(x))
    df_train.to_csv('../input/train_clean.csv', encoding='utf-8', index=False)

    df_test = pd.read_csv('../input/test_clean.csv', encoding='utf-8')
    df_test['comment_text'].apply(lambda x: clean(x))
    df_test.to_csv('../input/test_clean.csv', encoding='utf-8', index=False)
