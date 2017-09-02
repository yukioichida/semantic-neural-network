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

#logger.debug('often makes a very good meal of %s', 'visiting tourists')

from modules.input_data import prepare_input_data
from modules.datasets import SICKDataset

df = SICKDataset('asd.csv').data_frame()

input_data = prepare_input_data(df)

print("Max sentence length %s" % (input_data.max_sentence_length))
print(input_data.word_index)