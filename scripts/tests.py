
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
a = [1.2, 3.4, 4.6]
b = [2.0, 2.2, 3.0]

#print(stats.pearsonr(a, b))

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


from gensim.models.keyedvectors import KeyedVectors
WORD2VEC = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
a = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)

s = a.similarity('man','woman')

print(s)