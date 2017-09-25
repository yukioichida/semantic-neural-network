from modules.log_config import LOG
from modules.input_data import ProcessInputData
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import InputConfiguration, create_output

import datetime
import sys
from time import time

from keras.optimizers import Adadelta, Nadam, Adam, Adamax
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping


tensorboard = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True,
                          embeddings_freq=0, embeddings_metadata=True)
reduceLr = ReduceLROnPlateau()
early_stop = EarlyStopping(monitor='val_mean_absolute_error',patience=15)
callbacks = []

QUORA_FILE = 'C:\\dev_env\\ml\\datasets\\quora_questions_pair\\train.csv'
ALL_STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-processed.tsv'
STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts_all.txt'
STS_REDUCED_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-reduced.txt'
STS_2013 = 'C:\\dev_env\\ml\\datasets\\sts\\sts-2013.tsv'

SICK_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_complete.txt'
SICK_PRETRAIN_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\pretrain_sts.txt'
SICK_TRAIN_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_train.txt'
SICK_TEST_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_test.txt'

WORD2VEC = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
GLOVE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'
FAST_TEXT = 'C:\dev_env\ml\\datasets\\fasttext_english\\wiki.en.vec'

EMBEDDING_FILE = WORD2VEC
EMBEDDING_BINARY = EMBEDDING_FILE == WORD2VEC
EMBEDDING_DIM = 300 # dimension of the word embedding vectors
LSTM_HIDDEN_LAYERS = 50 # by the paper

BATCH_SIZE = 64
PRETRAIN_EPOCHS = 65
TRAIN_EPOCHS = 320

DROPOUT = 0.3
RECURRENT_DROPOUT = 0.3

PRETRAIN=True
TRAIN=True

pretrain_dataframe = STSDataset(SICK_PRETRAIN_FILE).data_frame()

dataset = SICKFullDataset(SICK_TRAIN_FILE)
sts_dataset = STSDataset(ALL_STS_FILE)
#sts_dataset = STSDataset(STS_FILE)
#sick_dataset = SICKDataset(SICK_TRAIN_FILE)
#sick_dataset = SICKFullDataset(SICK_TRAIN_FILE)

train_dataframe = pd.concat([dataset.data_frame(), sts_dataset.data_frame()], ignore_index=True)
#train_dataframe = dataset.data_frame()
LOG.info("DF size = %s" % (len(train_dataframe.index)))

test_df = SICKFullDataset(SICK_TEST_FILE).data_frame()

process = ProcessInputData()
pretrain_input, train_input, test_input = process.prepare_input_data(pretrain_dataframe, train_dataframe, test_df)

max_sentence_length = process.max_sentence_length
vocab_size = process.vocabulary_size + 1
word_index = process.word_index
LOG.info("Max Sentence Length %s | Vocab Size: %s" % (max_sentence_length, vocab_size))
#=======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
#=======================================
embedding_matrix = load_embedding_matrix(dataset.name(), EMBEDDING_FILE, word_index, binary=EMBEDDING_BINARY)

# =========================================
# ============= MODEL =====================
# =========================================
malstm = init_model(max_sentence_length, embedding_matrix, DROPOUT, RECURRENT_DROPOUT, vocab_size)
#gradient_clipping_norm = 0.95
gradient_clipping_norm = 1.5

lr = 1
# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(lr = lr, clipnorm=gradient_clipping_norm)
#optimizer = Adam()

malstm.compile(loss = 'mean_squared_error',
               optimizer=optimizer,
               metrics=['accuracy', 'mean_absolute_error'])

x1_test, x2_test, y_test = test_input.x1, test_input.x2, test_input.y
# =====================================
# ============= PRE TRAIN =============
# =====================================
import numpy as np
mae = np.float64(0)
if PRETRAIN:
    LOG.info("START PRE TRAIN")
    training_time = time()

    malstm.fit([pretrain_input.x1, pretrain_input.x2], pretrain_input.y,
               epochs= PRETRAIN_EPOCHS,  batch_size=BATCH_SIZE,
                callbacks=callbacks,
               #verbose = 2,
               validation_data=([x1_test, x2_test],y_test))

    print("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))

# =====================================
# ============= TRAIN =============
# =====================================
#x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(train_input_data.x1, train_input_data.x2, train_input_data.y,
#
#                                         test_size=0.2)
if TRAIN:
    x1_train, x2_train, y_train = train_input.x1, train_input.x2, train_input.y
    training_time = time()

    malstm.fit([x1_train, x2_train], y_train,
               epochs= TRAIN_EPOCHS,
               batch_size=BATCH_SIZE,
               #verbose=2,
               validation_data=([x1_test, x2_test], y_test),
               callbacks=callbacks
                )

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
                                  embedding_type="WORD2VEC",
                                  dropout=DROPOUT,
                                  recurrent_dropout=RECURRENT_DROPOUT)

create_output(y_pred, y_test, mae, input_config, obs=str(sys.argv))

