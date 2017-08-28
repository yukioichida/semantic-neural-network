#from gensim.models.keyedvectors import KeyedVectors

import os
import pandas as pd
import nltk
import numpy as np


# from keras.models import Model
# from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


WORD2VEC_FILE = "C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin"
GLOVE_FILE = "C:\dev_env\ml\datasets\glove.6B\glove.6B.300d.gensim.txt"

MAX_NB_WORDS = 60000
MAX_SENTENCE_LENGTH = 40 # maximum words in a sentence

DATASET_BASEDIR = "..\\..\\datasets\\"
# Paraphrase dataset
PP_DATASET = os.path.join(DATASET_BASEDIR, "pp\pp-unified-processed.tsv")
# STS english dataset
STS_DATASET = os.path.join(DATASET_BASEDIR, "similarity\en\sts-processed.tsv")
DELIMITER = '\t'
dataset_file = STS_DATASET
dataframe = pd.read_csv(dataset_file, delimiter='\t')

all_sentences = []
sentences_1 = []
sentences_2 = []
labels = []
for index, row in dataframe.iterrows():
    sentences_1.append(row['s1'])
    sentences_2.append(row['s2'])
    labels.append(row['label'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences_1)
tokenizer.fit_on_texts(sentences_2)

print('Found %s unique tokens.' % len(tokenizer.word_index))

input_sentences_1 = tokenizer.texts_to_sequences(sentences_1)
input_sentences_2 = tokenizer.texts_to_sequences(sentences_2)

x1 = pad_sequences(input_sentences_1, MAX_SENTENCE_LENGTH)
x2 = pad_sequences(input_sentences_1, MAX_SENTENCE_LENGTH)
y = np.array(labels)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state=42)



