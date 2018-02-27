# -*- coding:utf-8 -*-
"""
fast text model
https://arxiv.org/abs/1607.01759
"""


def create_ngrams_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngrams(sequences, token_indice, ngram_range=2):
    """
    add ngram features to sequence(input list)

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    :param sequences:
    :param token_indice:
    :param ngram_range:
    :return:
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list)-ngram_range+1):
            pass