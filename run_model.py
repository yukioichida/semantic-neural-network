from modules.log_config import LOG
from modules.input_data import ProcessInputData
from modules.datasets import *
from modules.model import init_model
from modules.embedding import load_embedding_matrix
from modules.result_data import InputConfiguration, create_output

import datetime
import sys
from sklearn.model_selection import train_test_split
from time import time

from keras.optimizers import Adadelta

QUORA_FILE = 'C:\\dev_env\\ml\\datasets\\quora_questions_pair\\train.csv'
STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-processed.tsv'
STS_REDUCED_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-reduced.txt'
STS_2013 = 'C:\\dev_env\\ml\\datasets\\sts\\sts-2013.tsv'

SICK_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_complete.txt'
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
PRETRAIN_EPOCHS = 1
TRAIN_EPOCHS = 1

pretrain_dataframe = STSDataset(STS_2013).data_frame()

#dataset = SICKFullDataset(SICK_TRAIN_FILE)
dataset = STSDataset(STS_FILE)
train_dataframe = dataset.data_frame()

test_df = SICKFullDataset(SICK_TEST_FILE).data_frame()

process = ProcessInputData()
pretrain_input_data, train_input_data = process.prepare_input_data(pretrain_dataframe, train_dataframe, rescaling_output=5)


max_sentence_length = train_input_data.max_sentence_length
print("Max Sentence Length %s" % (max_sentence_length))
test_sentences_1, test_sentences_2, test_labels = process.pre_process_data(test_df, invert=False)
process.tokenizer.fit_on_texts(test_sentences_1)
process.tokenizer.fit_on_texts(test_sentences_2)

word_index = process.tokenizer.word_index
vocab_size = len(word_index) + 1
LOG.info("Vocab size %s" % (vocab_size))
#=======================================
#   EMBEDDING MATRIX FOR WORD EMBEDDINGS
#=======================================
embedding_matrix = load_embedding_matrix(dataset.name(), EMBEDDING_FILE, word_index, binary=EMBEDDING_BINARY)

# =========================================
# ============= MODEL =====================
# =========================================
malstm = init_model(max_sentence_length, embedding_matrix, vocab_size)
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
    #x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(pretrain_input_data.x1,
    #                                                                         pretrain_input_data.x2,
    #                                                                         pretrain_input_data.y,
    #                                                                         test_size=0.2)
    training_time = time()
    malstm.fit([pretrain_input_data.x1, pretrain_input_data.x2], pretrain_input_data.y,
               epochs= PRETRAIN_EPOCHS,  batch_size=BATCH_SIZE)
    print("\nPré Training time finished.\n{} epochs in {}".format(PRETRAIN_EPOCHS, datetime.timedelta(seconds=time()-training_time)))

# =====================================
# ============= TRAIN =============
# =====================================
#x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(train_input_data.x1, train_input_data.x2, train_input_data.y,
#
#                                         test_size=0.2)
x1_train, x2_train, y_train = train_input_data.x1, train_input_data.x2, train_input_data.y
x1_test, x2_test, y_test = process.get_samples(test_sentences_1, test_sentences_2, test_labels, max_sentence_length, rescaling_output=5)
training_time = time()

from keras.callbacks import ReduceLROnPlateau

reduceLr = ReduceLROnPlateau(factor=0.5, patience=3)
malstm.fit([x1_train, x2_train], y_train,
           epochs= TRAIN_EPOCHS,
           batch_size=BATCH_SIZE,
           validation_data=([x1_test, x2_test], y_test),
           callbacks=[reduceLr])

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
                                  embedding_type="WORD2VEC")

create_output(y_pred, y_test, mae, input_config, obs=str(sys.argv))


with open('output_model.json', 'w') as json_file:
    json_file.write(malstm.to_json())
