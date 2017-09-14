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


class ProcessInputData:

    def __init__(self):
        self.tokenizer = Tokenizer()

    def pre_process_data(self,df, invert = True):
        sentences_1 = []
        sentences_2 = []
        labels = []
        for index, row in df.iterrows():
            sentences_1.append(prepare_text(row['s1']))
            sentences_2.append(prepare_text(row['s2']))
            label = float(row['label'])
            labels.append(round(label, 1))
            if invert:
                sentences_1.append(prepare_text(row['s2']))
                sentences_2.append(prepare_text(row['s1']))
                labels.append(round(label, 1))

        return (sentences_1, sentences_2, labels)


    def get_samples(self, sentences_1, sentences_2, label, max_sentence_length, rescaling_output=1):
        # Prepare the neural network inputs
        input_sentences_1 = self.tokenizer.texts_to_sequences(sentences_1)
        input_sentences_2 = self.tokenizer.texts_to_sequences(sentences_2)
        print("Vocab size on get samples %s" %(len(self.tokenizer.word_index)))
        x1 = pad_sequences(input_sentences_1, max_sentence_length)
        x2 = pad_sequences(input_sentences_2, max_sentence_length)
        y = np.array(label) / rescaling_output  # WARNING: LABEL RESCALING
        return (x1, x2, y)


    def prepare_input_data(self, pretrain_df, train_df, rescaling_output = 1) -> (InputData, InputData):
        train_sentences_1, train_sentences_2, train_labels = self.pre_process_data(train_df)
        pretrain_sentences_1, pretrain_sentences_2, pretrain_labels = self.pre_process_data(pretrain_df)

        self.tokenizer.fit_on_texts(train_sentences_1)
        self.tokenizer.fit_on_texts(train_sentences_2)
        self.tokenizer.fit_on_texts(pretrain_sentences_1)
        self.tokenizer.fit_on_texts(pretrain_sentences_2)

        word_index = self.tokenizer.word_index
        vocabulary_size = len(word_index)
        LOG.info("Vocabulary created. Size: %s", vocabulary_size)

        max_sentence_length = 0
        # The size of the input sequence is the size of the largest sequence of the input dataset
        for sentence_vec in [train_sentences_1, train_sentences_2, pretrain_sentences_1, pretrain_sentences_2]:
            for sentence in sentence_vec:
                sentence_length = len(sentence.split())
                if (sentence_length > max_sentence_length):
                    max_sentence_length = sentence_length

        # Prepare the neural network inputs
        (pretrain_x1, pretrain_x2, pretrain_y) = self.get_samples(pretrain_sentences_1, pretrain_sentences_2, pretrain_labels,
                                                     max_sentence_length, rescaling_output=rescaling_output)

        (train_x1, train_x2, train_y) = self.get_samples(train_sentences_1, train_sentences_2, train_labels,
                                                     max_sentence_length, rescaling_output=rescaling_output)

        pretrain_input_data =  InputData(word_index=word_index,
                         max_sentence_length = max_sentence_length,
                         vocab_size = vocabulary_size,
                         x1 = pretrain_x1,
                         x2 = pretrain_x2,
                         y = pretrain_y)

        train_input_data = InputData(word_index=word_index,
                                     max_sentence_length=max_sentence_length,
                                     vocab_size=vocabulary_size,
                                     x1=train_x1,
                                     x2=train_x2,
                                     y=train_y)

        return (pretrain_input_data, train_input_data)

