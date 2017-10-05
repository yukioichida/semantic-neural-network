# -*- coding: utf-8 -*-

BASE_PATH = '../datasets/'

SICK_FILE = BASE_PATH + 'sick_2014/SICK_complete.txt'
SICK_PRETRAIN_FILE = BASE_PATH + 'sick_2014/pretrain_sts.txt'
SICK_TRAIN_FILE = BASE_PATH + 'sick_2014/SICK_train.txt'
SICK_TRIAL_FILE = BASE_PATH + 'sick_2014/SICK_trial.txt'
SICK_TEST_FILE = BASE_PATH + 'sick_2014/SICK_test.txt'

WORD2VEC_FILE = BASE_PATH + 'word2vec/GoogleNews-vectors-negative300.bin'

GLOVE_FILE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'
FAST_TEXT_FILE = 'C:\dev_env\ml\\datasets\\fasttext_english\\wiki.en.vec'

EMBEDDING_FILE = WORD2VEC_FILE
EMBEDDING_NAME = "WORD2VEC"
EMBEDDING_BINARY = EMBEDDING_FILE == WORD2VEC_FILE
EMBEDDING_DIM = 300  # dimension of the word embedding vectors"

LSTM_HIDDEN_LAYERS = 50  # by the paper

BATCH_SIZE = 32
PRETRAIN_EPOCHS = 50
TRAIN_EPOCHS = 320

DROPOUT = 0.0
RECURRENT_DROPOUT = 0.2

PRETRAIN = True
TRAIN = True

REMOVE_STOPWORDS = False

FIT_VERBOSE = 2
