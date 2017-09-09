from modules.log_config import LOG
from modules.input_data import prepare_input_data
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import InputConfiguration, create_output

import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from time import time

from gensim.models.keyedvectors import KeyedVectors

from keras.optimizers import Adadelta

EMBEDDING_DIM = 300 # dimension of the word embedding vectors
LSTM_HIDDEN_LAYERS=50 # by the paper
BATCH_SIZE = 32

QUORA_FILE = 'C:\\dev_env\\ml\\datasets\\quora_questions_pair\\train.csv'
STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-processed.tsv'
STS_REDUCED_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-reduced.txt'
SICK_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_complete.txt'

WORD2VEC = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
GLOVE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'
FAST_TEXT = 'C:\dev_env\ml\\datasets\\fasttext_english\\wiki.en.vec'

EMBEDDING_FILE = FAST_TEXT
EMBEDDING_BINARY = EMBEDDING_FILE == WORD2VEC

PRETRAIN_EPOCHS = 60
TRAIN_EPOCHS = 360

pretrain_dataframe = SICKDataset(SICK_FILE).data_frame()
train_dataframe = STSDataset(STS_FILE).data_frame()

pretrain_input_data, train_input_data = prepare_input_data(pretrain_dataframe, train_dataframe, rescaling_output=5)

max_sentence_length = train_input_data.max_sentence_length
vocab_size = train_input_data.vocab_size + 1
word_index = train_input_data.word_index

#=======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
#=======================================
embedding_matrix = load_embedding_matrix("sts", EMBEDDING_FILE, word_index, binary=EMBEDDING_BINARY)
# =========================================
# ============= MODEL =====================
# =========================================
malstm = init_model(BATCH_SIZE, max_sentence_length, embedding_matrix, vocab_size)
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
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(pretrain_input_data.x1,
                                                                             pretrain_input_data.x2,
                                                                             pretrain_input_data.y,
                                                                             test_size=0.2)
    training_time = time()
    malstm.fit([pretrain_input_data.x1, pretrain_input_data.x2], pretrain_input_data.y,
               epochs= PRETRAIN_EPOCHS,  batch_size=BATCH_SIZE)
    print("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))

# =====================================
# ============= TRAIN =============
# =====================================
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(train_input_data.x1, train_input_data.x2, train_input_data.y,
                                                                         test_size=0.2)
training_time = time()
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
y_pred = malstm.predict([x1_test, x2_test])
input_config = InputConfiguration(batch_size=BATCH_SIZE,
                                  pretrain_epoch=PRETRAIN_EPOCHS,
                                  epoch=TRAIN_EPOCHS,
                                  embedding_type="FAST_TEXT")

create_output(y_pred, y_test, mae, input_config)

