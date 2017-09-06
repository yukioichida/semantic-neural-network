from modules.log_config import LOG
from modules.input_data import prepare_input_data
from modules.datasets import *

import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from time import time

from gensim.models.keyedvectors import KeyedVectors

from keras.optimizers import Adadelta
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Merge, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K

EMBEDDING_DIM = 300 # dimension of the word embedding vectors
LSTM_HIDDEN_LAYERS=50 # by the paper
BATCH_SIZE = 64

QUORA_FILE = 'C:\\dev_env\\ml\\datasets\\quora_questions_pair\\train.csv'
#STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts_all.txt'
STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-processed.tsv'
STS_REDUCED_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-reduced.txt'

SICK_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_complete.txt'

WORD2VEC = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
GLOVE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'
FAST_TEXT = 'C:\dev_env\ml\\datasets\\fasttext_english\\wiki.en.vec'

EMBEDDING_FILE = GLOVE
EMBEDDING_BINARY = EMBEDDING_FILE == WORD2VEC

pretrain_dataframe = SICKDataset(SICK_FILE).data_frame()
#print(pretrain_dataframe)
#pretrain_dataframe = STSDataset(STS_REDUCED_FILE).data_frame()
train_dataframe = STSDataset(STS_FILE).data_frame()

pretrain_input_data, train_input_data = prepare_input_data(pretrain_dataframe, train_dataframe, rescaling_output=5)

max_sentence_length = train_input_data.max_sentence_length
vocab_size = train_input_data.vocab_size + 1
word_index = train_input_data.word_index

#=======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
#=======================================
LOG.info('Loading embedding model from %s', EMBEDDING_FILE)
embedding_model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=EMBEDDING_BINARY)
#embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
embedding_matrix = 1 * np.random.randn(vocab_size, EMBEDDING_DIM)  # This will be the embedding matrix
embedding_matrix[0] = 0  # So that the padding will be ignored
LOG.info('Creating the embedding matrix')
for word, idx in word_index.items():
    if idx >= vocab_size:
        continue
    if word in embedding_model.vocab:
        embedding_vector = embedding_model.word_vec(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

LOG.info('Embedding matrix as been created, removing embedding model from memory')
del embedding_model
# =========================================
# ============= MODEL =====================
# =========================================

LOG.info("Creating model...")

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
base_lstm = LSTM(LSTM_HIDDEN_LAYERS)

left_representation = base_lstm(left_encoder)
right_representation = base_lstm(right_encoder)

def out_shape(shapes):
    print("out_shape")
    print(shapes)
    return (None, 1)

def exponent_neg_manhattan_distance(vector):
#    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    result = K.exp(-K.sum(K.abs(vector[0]-vector[1]), axis=1, keepdims=True))
    print("Lambda merge: %s" % (result))
    return result

malstm_distance = Lambda(exponent_neg_manhattan_distance, output_shape=out_shape)([left_representation, right_representation])

malstm = Model([left_input, right_input], [malstm_distance])
gradient_clipping_norm = 1.25
# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss = 'mean_squared_error',
               optimizer=optimizer,
               metrics=['accuracy', 'mean_absolute_error'])

# =====================================
# ============= PRE TRAIN =============
# =====================================
pre_train = True
if pre_train:
    LOG.info("START PRE TRAIN")
    #x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(pretrain_input_data.x1, pretrain_input_data.x2, pretrain_input_data.y,
    #                                                                        test_size=0.2)
    training_time = time()
    PRETRAIN_EPOCHS = 60

    malstm.fit([pretrain_input_data.x1, pretrain_input_data.x2], pretrain_input_data.y,
               epochs= PRETRAIN_EPOCHS,
               batch_size=32)

    print("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))


    #score, acc, mae = malstm.evaluate([x1_test, x2_test], y_test, batch_size=BATCH_SIZE)
    #print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))

# =====================================
# ============= TRAIN =============
# =====================================
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(train_input_data.x1, train_input_data.x2, train_input_data.y,
                                                                         test_size=0.2)
training_time = time()
TRAIN_EPOCHS = 300
malstm.fit([x1_train, x2_train], y_train,
           epochs= TRAIN_EPOCHS,
           batch_size=BATCH_SIZE,
           validation_data=([x1_test, x2_test], y_test))

print("\nTraining time finished.\n{} epochs in {}".format(TRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))

score, acc, mae = malstm.evaluate([x1_test, x2_test], y_test, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))

# =====
# PREDICT
# =====
import scipy.stats as stats
from sklearn.metrics import mean_squared_error as mse


y_pred = malstm.predict([x1_test, x2_test])
print(y_pred.ravel())
print(y_test)
pr = stats.pearsonr(y_pred.ravel()*5, y_test*5)[0]
sr = stats.spearmanr(y_pred.ravel()*5, y_test*5)[0]
e = mse(y_pred.ravel()*5, y_test*5)
print('Embedding: '+EMBEDDING_FILE)
print('RESULTS: %s PRETRAIN EPOCH, %s TRAIN EPOCH' % (PRETRAIN_EPOCHS, TRAIN_EPOCHS))
print(' Pearson: %f' % (pr))
print(' Spearman: %f' % ( sr))
print(' MSE: %f' % ( e))

