from modules.log_config import LOG
from modules.input_data import prepare_input_data
from modules.datasets import STSDataset

from gensim.models.keyedvectors import KeyedVectors

import numpy as np
from time import time
import datetime

from keras.optimizers import Adadelta
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Merge
from keras.layers.recurrent import GRU, LSTM
from keras.layers.merge import concatenate
from sklearn.model_selection import train_test_split
from keras import backend as K

MAX_NB_WORDS = 60000
EMBEDDING_DIM = 300 # dimension of the word embedding vectors
LSTM_HIDDEN_LAYERS=50 # by the paper
BATCH_SIZE = 100

QUORA_FILE = 'C:\\dev_env\\ml\\datasets\\quora_questions_pair\\train.csv'
STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts_all.txt'
SICK_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_complete.txt'
WORD2VEC = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
GLOVE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'

EMBEDDING_FILE = WORD2VEC

dataframe = STSDataset(STS_FILE).data_frame()
input_data = prepare_input_data(dataframe, rescaling_output=5)

max_sentence_length = input_data.max_sentence_length
vocab_size = input_data.vocab_size
word_index = input_data.word_index





def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

LOG.info("Creating model...")

# ============= MODEL =====================
# A entrada recebe os índices das palavras no vocabulário, para fazer o lookup na tabela de embeddings
left_input = Input(shape=(max_sentence_length,), dtype='int32')
right_input = Input(shape=(max_sentence_length,), dtype='int32')

#Camada de embedding
embedding_layer = Embedding(vocab_size, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_sentence_length,
                            trainable=False)

left_encoder = embedding_layer(left_input)
right_encoder = embedding_layer(right_input)

# LSTM
base_lstm = GRU(LSTM_HIDDEN_LAYERS)

left_representation = base_lstm(left_encoder)
right_representation = base_lstm(right_encoder)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))\
    ([left_representation, right_representation])

malstm = Model([left_input, right_input], [malstm_distance])
gradient_clipping_norm = 1.25
# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss = 'mean_squared_error',
               optimizer=optimizer,
               metrics=['accuracy'])

training_time = time()
EPOCHS = 200
malstm.fit([x1_train, x2_train], y_train,
           epochs= EPOCHS,
           batch_size=BATCH_SIZE,
           validation_data=([x1_test, x2_test], y_test))

print("\nTraining time finished.\n{} epochs in {}".format(EPOCHS, datetime.timedelta(seconds=time()-training_time)))

score, acc = malstm.evaluate([x1_test, x2_test], y_test, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
