
QUORA_FILE = 'C:\\dev_env\\ml\\datasets\\quora_questions_pair\\train.csv'
ALL_STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-processed.tsv'
STS_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts_all.txt'
STS_REDUCED_FILE = 'C:\\dev_env\\ml\\datasets\\sts\\sts-reduced.txt'

SICK_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_complete.txt'
SICK_PRETRAIN_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\pretrain_sts.txt'
SICK_TRAIN_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_train.txt'
SICK_TEST_FILE = 'C:\\dev_env\ml\\datasets\\sick_2014\\SICK_test.txt'

WORD2VEC_FILE = 'C:\dev_env\ml\datasets\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
GLOVE_FILE = 'C:\dev_env\ml\datasets\glove.6B\\glove.6B.300d.gensim.txt'
FAST_TEXT_FILE = 'C:\dev_env\ml\\datasets\\fasttext_english\\wiki.en.vec'

EMBEDDING_FILE = WORD2VEC_FILE
EMBEDDING_BINARY = EMBEDDING_FILE == WORD2VEC_FILE
EMBEDDING_DIM = 300 # dimension of the word embedding vectors
LSTM_HIDDEN_LAYERS = 50 # by the paper

BATCH_SIZE = 32
PRETRAIN_EPOCHS = 50
TRAIN_EPOCHS = 320

DROPOUT = 0.35
RECURRENT_DROPOUT = 0.35

PRETRAIN=False
TRAIN=True