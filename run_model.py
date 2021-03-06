# -*- coding: utf-8 -*-
import datetime
import os
import sys
from time import time

from keras.optimizers import Adadelta

from modules.log_config import LOG
from modules.configs import *
from modules.input_data import ProcessInputData
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import create_output
from modules.plot_result import plot_fit_history, save_history

#pretrain_df = STSDataset(SICK_PRETRAIN_FILE).data_frame()
train_df = SICKFullDataset(SICK_TRAIN_FILE).data_frame()
# train_df = STSDataset(SICK_AUGMENTED_FILE).data_frame()
# train_df = STSDataset(SICK_AUGMENTED_NOUN_FILE).data_frame()
val_df = SICKFullDataset(SICK_TRIAL_FILE).data_frame()
test_df = SICKFullDataset(SICK_TEST_FILE).data_frame()


train_df = pd.concat([train_df, val_df])

# ===============================
# PREPARE INPUT DATA - PREPROCESS
# ======================
process = ProcessInputData()
train_data = process.prepare_data([train_df], 'SICK')
train_input = train_data[0]
test_input = process.prepare_data([test_df], 'SICK')
max_sentence_length = process.max_sentence_length
vocab_size = process.vocabulary_size + 1
word_index = process.word_index
LOG.info("Max Sentence Length %s | Vocab Size: %s" % (max_sentence_length, vocab_size))

# =======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
# =======================================
print EMBEDDING_BINARY
embedding_matrix = load_embedding_matrix("SICK", word_index)

# =========================================
#     CREATING MODEL
# =========================================
model = init_model(max_sentence_length, embedding_matrix, DROPOUT, RECURRENT_DROPOUT, vocab_size)
gradient_clipping_norm = 1.6
optimizer = Adadelta(lr=LR, clipnorm=gradient_clipping_norm)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# =========================================
# ============== TRAIN MODEL ==============
# =========================================
start_train = time()
# ============= PRE TRAIN =============
#if PRETRAIN:
#    LOG.info("START PRE TRAIN")
#    training_time = time()

#    model.fit([pretrain_input.x1, pretrain_input.x2], pretrain_input.y,
#              epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE,
#              validation_split=0.3, verbose=FIT_VERBOSE)

#    duration = datetime.timedelta(seconds=time() - training_time)
#    LOG.info("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, duration))
# ============= TRAIN =============
LOG.info("START TRAIN")
training_time = time()
train_history = model.fit([train_input.x1, train_input.x2], train_input.y,
                          epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE,
                          validation_split=0.1, verbose=FIT_VERBOSE)

duration = datetime.timedelta(seconds=time() - training_time)
LOG.info("\nTraining time finished.\n{} epochs in {}".format(TRAIN_EPOCHS, duration))
total_duration = datetime.timedelta(seconds=time() - start_train)

# ======= STORE TRAINED MODEL ======
LOG.info('Saving Model')
timestamp = (int(round(time() * 1000)))
model_file = 'model_%s.h5' % timestamp
model_file_path = os.path.join(RESULTS_DIR, model_file)
model.save(model_file_path)
# ========================
#          PREDICT
# ========================
y_pred = model.predict([test_input.x1, test_input.x2])

# =========================
# CREATE OUTPUT FILE
# =========================
LOG.info('Creating result file')
create_output(y_pred, test_input.y, total_duration, model_file, timestamp, obs=str(sys.argv))

# =========================
# ===== PLOT GRAPHICS =====
# =========================
LOG.info('Ploting results')
if len(sys.argv) == 2:
    title = str(sys.argv[1])
else:
    title = 'Custo X Épocas'
# plot_fit_history(train_history, model_file, title)
save_history(train_history, model_file)
