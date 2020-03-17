# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:58:29 2019

@author: HPH
"""
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
            
max_length = 10

results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[: max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1


import string

characters = string.printable

token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50

results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

for i, sample in enumerate(samples):
    for j, character in list(enumerate(sample))[: max_length]:
        index = token_index[character]
        results[i, j, index] = 1