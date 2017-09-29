# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from modules.prepare_text import prepare_text

'''
    Classes and components
'''

class InputData:

    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y


class ProcessInputData:

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.max_sentence_length = -1


    def pre_process_data(self,df):
        sentences_1 = []
        sentences_2 = []
        labels = []
        for index, row in df.iterrows():
            sentences_1.append(prepare_text(row['s1']))
            sentences_2.append(prepare_text(row['s2']))
            label = float(row['label'])
            labels.append(round(label, 1))

        return sentences_1, sentences_2, labels


    def get_samples(self, sentences_1, sentences_2, label, rescaling_output=1):
        # Prepare the neural network inputs
        input_sentences_1 = self.tokenizer.texts_to_sequences(sentences_1)
        input_sentences_2 = self.tokenizer.texts_to_sequences(sentences_2)
        x1 = pad_sequences(input_sentences_1, self.max_sentence_length)
        x2 = pad_sequences(input_sentences_2, self.max_sentence_length)
        y = np.array(label)
        y = np.clip(y, 1, 5) # paper definition
        y = (y - 1) / 4  # WARNING: LABEL RESCALING
        return x1, x2, y


    def prepare_input_data(self, pretrain_df, train_df, test_df):
        train_sentences_1, train_sentences_2, train_labels = self.pre_process_data(train_df)
        pretrain_sentences_1, pretrain_sentences_2, pretrain_labels = self.pre_process_data(pretrain_df)
        test_sentences_1, test_sentences_2, test_labels = self.pre_process_data(test_df)

        self.tokenizer.fit_on_texts(train_sentences_1)
        self.tokenizer.fit_on_texts(train_sentences_2)
        self.tokenizer.fit_on_texts(pretrain_sentences_1)
        self.tokenizer.fit_on_texts(pretrain_sentences_2)
        self.tokenizer.fit_on_texts(test_sentences_1)
        self.tokenizer.fit_on_texts(test_sentences_2)

        self.word_index = self.tokenizer.word_index
        self.vocabulary_size = len(self.word_index)

        max_sentence_length = 0
        # The size of the input sequence is the size of the largest sequence of the input dataset
        for sentence_vec in [train_sentences_1, train_sentences_2, pretrain_sentences_1, pretrain_sentences_2]:
            for sentence in sentence_vec:
                sentence_length = len(sentence.split())
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length

        self.max_sentence_length = max_sentence_length

        # Prepare the neural network inputs
        (pretrain_x1, pretrain_x2, pretrain_y) = self.get_samples(pretrain_sentences_1, pretrain_sentences_2, pretrain_labels)

        (train_x1, train_x2, train_y) = self.get_samples(train_sentences_1, train_sentences_2, train_labels)

        (test_x1, test_x2, test_y) = self.get_samples(test_sentences_1, test_sentences_2, test_labels)

        pretrain_input_data = InputData(x1=pretrain_x1,
                                         x2=pretrain_x2,
                                         y=pretrain_y)

        train_input_data = InputData(x1=train_x1,
                                     x2=train_x2,
                                     y=train_y)

        test_input_data = InputData(x1=test_x1,
                                    x2=test_x2,
                                    y=test_y)

        return pretrain_input_data, train_input_data, test_input_data
