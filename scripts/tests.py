'''
building the vocabulary

from keras.preprocessing.text import Tokenizer

text = ["focus on the work keras", "Test in the keras"]
text2 = ["ths is a second collection", "I need to write english soon as possible keras"]

tokenizer = Tokenizer(100)
tokenizer.fit_on_texts(text)
tokenizer.fit_on_texts(text2)

#print(tokenizer.num_words)
print(tokenizer.texts_to_sequences(text2))
print(tokenizer.texts_to_sequences(text))
print('Found %s unique tokens.' % len(tokenizer.word_index))
'''

'''
split test and train sets


import numpy as np
from sklearn.model_selection import train_test_split

a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
y = ['a', 'b', 'c', 'd']

Xtrain, Xtest, X2train, X2test, Ytrain, Ytest = train_test_split(np.array(a), np.array(b), np.array(y), shuffle = False)

print(Xtrain)
print(Xtest)
print(X2train)
print(X2test)
print(Ytrain)
print(Ytest)
'''
#import nltk
#from nltk.corpus import wordnet as wn

sent = "The girl, who is little, is combing her hair into a pony tail."

#for word in nltk.word_tokenize(sent):

#word = wn.synset('little.a.01')

#print(word.hypernyms())

import pandas as pd

df = pd.read_csv('asd.csv', encoding='utf-8', sep='\t')

#print(df)

#print(df[df.entailment_judgment != 'CONTRADICTION'])

sentences = df['sentence_A']

#print(sentences.values)
#import numpy as np
#embeddings = 1 * np.random.randn(10 + 1, 3)  # This will be the embedding matrix
#embeddings[0] = 0  # So that the padding will be ignored
#print(embeddings)

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s : %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                       index=[0, 1, 2, 3])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                     'D': ['D2', 'D3', 'D6', 'D7'],
                     'F': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])

result = pd.concat([df1, df4], axis=1)
#print(result)

import numpy as np
def sts_labels2categorical(labels, nclass=6):
    """
    From continuous labels in [0,5], generate 5D binary-ish vectors.
    This enables us to do classification instead of regression.
    (e.g. sigmoid output would be troublesome with the original labeling)
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    (Based on https://github.com/ryankiros/skip-thoughts/blob/master/eval_sick.py)
    """
    Y = np.zeros((len(labels), nclass))
    for j, y in enumerate(labels):
        if np.floor(y) + 1 < nclass:
            Y[j, int(np.floor(y)) + 1] = y - np.floor(y)
        Y[j, int(np.floor(y))] = np.floor(y) - y + 1
    return Y


#print (sts_labels2categorical([3.2, 4.5, 1.2]))

import scipy.stats as stats
array = [[1],[2],[3], [4.5]]
print (np.array(array).ravel())
#print(stats.pearsonr([1, 2, 3], [0.8, 1.9, 3.2]))








