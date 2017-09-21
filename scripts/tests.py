
import os
from pathlib import Path

dir = os.path.dirname(__file__)

print (Path(dir).parent)
#print(stats.pearsonr([1, 2, 3], [0.8, 1.9, 3.2]))

import time
milli_sec = int(round(time.time() * 1000))
print(milli_sec)


a = 3.7889
b = (a - 1) / 4

print (b)



b = a / 5
#print(b)

import scipy.stats as stats
a = [1, 2, 3]
b = [1, 5, 7]

print(stats.pearsonr(a, b))
print(stats.pearsonr(b, a))

a = a * 4
b = b * 4

print(stats.pearsonr(a, b))

'''
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

a = "teste hello world"
b = "hello word token test"
t = Tokenizer()
t.fit_on_texts([a])
t.fit_on_texts([b])

print(t.word_index)
xs = t.texts_to_sequences([a, b])
xsd = pad_sequences(xs, 4)
print(xsd)
'''

'''
from gensim.models.keyedvectors import KeyedVectors
WORD2VEC = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
a = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)

s = a.similarity('man','woman')

print(s)
'''

''''
pred = [2.4297, 3.02507,3.11535, 4.12365, 3.87481, 3.37664, 2.25154, 2.26633, 2.42719, 4.34101, 4.34101, 2.91809, 4.28756, 4.00735, 1.47126, 1.61245, 3.63899, 2.85903, 3.55474,0]
gt = [3.3, 3.7, 3.0, 4.9, 3.7, 3.3, 2.7, 2.9, 2.3, 4.9, 3.6, 3.0, 4.3, 4.1, 3.2, 3.3,4.0,4.2, 4.7, 4.5]
print(stats.pearsonr(pred, gt))
'''


import numpy as np
a = np.array([1, 2, 3, 4, 5])
#print ((a-1)/4)