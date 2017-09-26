from modules.log_config import LOG
from modules.configs import *
from modules.input_data import ProcessInputData
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import InputConfiguration, create_output

import datetime
import sys
import numpy as np
from time import time

from keras.optimizers import Adadelta

callbacks = []

pretrain_dataframe = STSDataset(SICK_PRETRAIN_FILE).data_frame()
train_dataframe = SICKFullDataset(SICK_TRAIN_FILE).data_frame()
test_df = SICKFullDataset(SICK_TEST_FILE).data_frame()
LOG.info("Train size = %s" % (len(train_dataframe.index)))

#======================
# PREPARE INPUT DATA
#======================
process = ProcessInputData()
pretrain_input, train_input, test_input = process.prepare_input_data(pretrain_dataframe, train_dataframe, test_df)

max_sentence_length = process.max_sentence_length
vocab_size = process.vocabulary_size + 1
word_index = process.word_index
LOG.info("Max Sentence Length %s | Vocab Size: %s" % (max_sentence_length, vocab_size))
#=======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
#=======================================
embedding_matrix = load_embedding_matrix("WORD2VEC", EMBEDDING_FILE, word_index, binary=EMBEDDING_BINARY)
# =========================================
#     MAIN MODEL
# =========================================
malstm = init_model(max_sentence_length, embedding_matrix, DROPOUT, RECURRENT_DROPOUT, vocab_size)
gradient_clipping_norm = 1.5
lr = 1
optimizer = Adadelta(lr = lr, clipnorm=gradient_clipping_norm)
malstm.compile(loss = 'mean_squared_error',
               optimizer=optimizer,
               metrics=['accuracy', 'mean_absolute_error'])

x1_test, x2_test, y_test = test_input.x1, test_input.x2, test_input.y
# =====================================
# ============= PRE TRAIN =============
# =====================================
mae = np.float64(0)
if PRETRAIN:
    LOG.info("START PRE TRAIN")
    training_time = time()

    malstm.fit([pretrain_input.x1, pretrain_input.x2], pretrain_input.y,
               epochs= PRETRAIN_EPOCHS,  batch_size=BATCH_SIZE,
                callbacks=callbacks,
               validation_data=([x1_test, x2_test],y_test))

    print("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))

# =================================
# ============= TRAIN =============
# =================================
if TRAIN:
    x1_train, x2_train, y_train = train_input.x1, train_input.x2, train_input.y
    training_time = time()

    malstm.fit([x1_train, x2_train], y_train,
               epochs= TRAIN_EPOCHS,
               batch_size=BATCH_SIZE,
               validation_data=([x1_test, x2_test], y_test),
               callbacks=callbacks)

    print("\nTraining time finished.\n{} epochs in {}".format(TRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))
    score, acc, mae = malstm.evaluate([x1_test, x2_test], y_test, batch_size=BATCH_SIZE)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))

# ================
# PREDICT
# ================
y_pred = malstm.predict([x1_test, x2_test])
input_config = InputConfiguration(batch_size=BATCH_SIZE,
                                  pretrain_epoch=PRETRAIN_EPOCHS,
                                  epoch=TRAIN_EPOCHS,
                                  embedding_type="WORD2VEC",
                                  dropout=DROPOUT,
                                  recurrent_dropout=RECURRENT_DROPOUT)

create_output(y_pred, y_test, mae, input_config, obs=str(sys.argv))

