from gensim.models.keyedvectors import KeyedVectors

import os
import pandas as pd
import nltk
import numpy as np
from time import time
import datetime


from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Merge
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


WORD2VEC_FILE = "C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin"
GLOVE_FILE = "C:\dev_env\ml\datasets\glove.6B\glove.6B.300d.gensim.txt"

MAX_NB_WORDS = 60000
MAX_SENTENCE_LENGTH = 40 # maximum words in a sentence
EMBEDDING_DIM = 300 # dimension of the word embedding vectors

LSTM_HIDDEN_LAYERS=50 # by the paper
BATCH_SIZE = 100

DATASET_BASEDIR = "..\\..\\datasets\\"
# Paraphrase dataset
PP_DATASET = os.path.join(DATASET_BASEDIR, "pp\pp-unified-processed.tsv")
# STS english dataset
STS_DATASET = os.path.join(DATASET_BASEDIR, "similarity\en\sts-processed.tsv")
DELIMITER = '\t'
dataset_file = STS_DATASET

dataframe = pd.read_csv(dataset_file, delimiter='\t')
dataframe[['label']] = dataframe[['label']].astype(float)
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

# Prepare embedding matrix for the lookup of the embedding layer
word_index = tokenizer.word_index
num_words = min(MAX_NB_WORDS, len(word_index))
print('Found %s unique tokens. Num. words used %s' % (len(word_index), num_words))

embedding_matrix = np.loadtxt("embedding_matrix.txt")

# Prepare the neural network inputs
input_sentences_1 = tokenizer.texts_to_sequences(sentences_1)
input_sentences_2 = tokenizer.texts_to_sequences(sentences_2)

x1 = pad_sequences(input_sentences_1, MAX_SENTENCE_LENGTH)
x2 = pad_sequences(input_sentences_2, MAX_SENTENCE_LENGTH)
y = np.array(labels)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, test_size=0.2, random_state=42)



def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# ============= MODEL =====================
# A entrada recebe os índices das palavras no vocabulário, para fazer o lookup na tabela de embeddings
left_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
right_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')

#Camada de embedding
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENTENCE_LENGTH,
                            trainable=False)

left_encoder = embedding_layer(left_input)
right_encoder = embedding_layer(right_input)

# LSTM
base_lstm = LSTM(LSTM_HIDDEN_LAYERS)

left_representation = base_lstm(left_encoder)
right_representation = base_lstm(right_encoder)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))\
    ([left_representation, right_representation])

malstm = Model([left_input, right_input], [malstm_distance])

malstm.compile(loss = 'mean_squared_error',
               optimizer='nadam',
               metrics=['acc'])

training_time = time()
EPOCHS = 2
malstm.fit([x1_train, x2_train], y_train,
           nb_epoch= EPOCHS,
           batch_size=BATCH_SIZE,
           validation_data=([x1_test, x2_test], y_test))

print("Training time finished.\n{} epochs in {}".format(EPOCHS, datetime.timedelta(seconds=time()-training_time)))

score, acc = malstm.evaluate([x1_test, x2_test], y_test, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))
