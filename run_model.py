# -*- coding: utf-8 -*-
from modules.log_config import LOG
from modules.configs import *
from modules.input_data import ProcessInputData
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import create_output

import datetime
import os
import sys
import numpy as np
from time import time

from keras.optimizers import Adadelta

callbacks = []

pretrain_df = STSDataset(SICK_PRETRAIN_FILE).data_frame()
# train_df = SICKFullDataset(SICK_TRAIN_FILE).data_frame()

train_df = STSDataset(SICK_AUGMENTED_FILE).data_frame()
val_df = SICKFullDataset(SICK_TRIAL_FILE).data_frame()
train_df = pd.concat([train_df, val_df])
test_df = SICKFullDataset(SICK_TEST_FILE).data_frame()
LOG.info("Train size = %s" % (len(train_df.index)))

# ======================
# PREPARE INPUT DATA
# ======================
process = ProcessInputData()
pretrain_input, train_input, test_input = process.prepare_input_data(pretrain_df, train_df, test_df)

max_sentence_length = process.max_sentence_length
vocab_size = process.vocabulary_size + 1
word_index = process.word_index
LOG.info("Max Sentence Length %s | Vocab Size: %s" % (max_sentence_length, vocab_size))
# =======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
# =======================================
embedding_matrix = load_embedding_matrix("SICK", word_index)
# =========================================
#     MAIN MODEL
# =========================================
malstm = init_model(max_sentence_length, embedding_matrix, DROPOUT, RECURRENT_DROPOUT, vocab_size)
gradient_clipping_norm = 1.6
optimizer = Adadelta(lr=LR, clipnorm=gradient_clipping_norm)
malstm.compile(loss='mean_squared_error',
               optimizer=optimizer)

# =========================================
# ============== TRAIN MODEL ==============
# =========================================
start_train = time()
# ============= PRE TRAIN =============
mae = np.float64(0)
if PRETRAIN:
    LOG.info("START PRE TRAIN")
    training_time = time()

    malstm.fit([pretrain_input.x1, pretrain_input.x2], pretrain_input.y,
               epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE,
               callbacks=callbacks,
               validation_split=0.1,
               verbose=FIT_VERBOSE)

    duration = datetime.timedelta(seconds=time() - training_time)
    print("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, duration))

# ============= TRAIN =============
if TRAIN:
    LOG.info("START TRAIN")
    training_time = time()

    malstm.fit([train_input.x1, train_input.x2], train_input.y,
               epochs=TRAIN_EPOCHS,
               batch_size=BATCH_SIZE,
               validation_split=0.1,
               callbacks=callbacks,
               verbose=FIT_VERBOSE)

    duration = datetime.timedelta(seconds=time() - training_time)
    print("\nTraining time finished.\n{} epochs in {}".format(TRAIN_EPOCHS, duration))

total_duration = datetime.timedelta(seconds=time() - start_train)

# ======= STORE TRAINED MODEL ======
model_file = 'model_%s.h5' % (int(round(time() * 1000)))
model_file_path = os.path.join(SAVED_MODEL_DIR, model_file)
malstm.save(model_file_path)

# ========================
#          PREDICT
# ========================
y_pred = malstm.predict([test_input.x1, test_input.x2])

# =========================
# CREATE OUTPUT FILE
# =========================
create_output(y_pred, test_input.y, mae, total_duration, model_file, obs=str(sys.argv))
