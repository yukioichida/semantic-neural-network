from modules.prepare_text import prepare_text
from modules.log_config import LOG
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

'''
    Classes and components
'''

class InputData:

    def __init__(self, word_index, vocab_size, max_sentence_length, x1, x2, y):
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.word_index = word_index


def prepare_input_data(dataframe, rescaling_output = 1):
    sentences_1 = []
    sentences_2 = []
    labels = []
    for index, row in dataframe.iterrows():
        sentences_1.append(prepare_text(row['s1']))
        sentences_2.append(prepare_text(row['s2']))
        labels.append(float(row['label']))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_1)
    tokenizer.fit_on_texts(sentences_2)

    word_index = tokenizer.word_index
    vocabulary_size = len(word_index)
    LOG.info("Vocabulary created. Size: %s", vocabulary_size)

    # Prepare the neural network inputs
    input_sentences_1 = tokenizer.texts_to_sequences(sentences_1)
    input_sentences_2 = tokenizer.texts_to_sequences(sentences_2)

    max_sentence_length = 0
    # The size of the input sequence is the size of the largest sequence of the input dataset
    for sentence_vec in [sentences_1, sentences_2]:
        for sentence in sentence_vec:
            sentence_length = len(sentence.split())
            if (sentence_length > max_sentence_length):
                max_sentence_length = sentence_length

    x1 = pad_sequences(input_sentences_1, max_sentence_length)
    x2 = pad_sequences(input_sentences_2, max_sentence_length)
    # WARNING: LABEL RESCALING
    y = np.array(labels) / rescaling_output

    return InputData(word_index=word_index,
                     max_sentence_length = max_sentence_length,
                     vocab_size = vocabulary_size,
                     x1 = x1,
                     x2 = x2,
                     y = y)