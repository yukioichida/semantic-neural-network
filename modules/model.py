from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K
from keras.initializers import random_uniform

def init_model(batch_size, max_sequence_length, embedding_matrix,
               vocab_size, lstm_hidden_layers=50, embedding_dim = 300) -> Model:

    # A entrada recebe os índices das palavras no vocabulário, para fazer o lookup na tabela de embeddings
    left_input = Input(shape=(max_sequence_length,), dtype='int32')
    right_input = Input(shape=(max_sequence_length,), dtype='int32')

    # Camada de embedding
    embedding_layer = Embedding(vocab_size, embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    left_encoder = embedding_layer(left_input)
    right_encoder = embedding_layer(right_input)

    # LSTM
    bias_initializer = random_uniform(minval=-0.5, maxval=0.5)
    base_lstm = LSTM(lstm_hidden_layers, bias_initializer=bias_initializer)

    left_representation = base_lstm(left_encoder)
    right_representation = base_lstm(right_encoder)

    def out_shape(shapes):
        return (None, 1)

    def exponent_neg_manhattan_distance(vector):
        #    ''' Helper function for the similarity estimate of the LSTMs outputs'''
        return K.exp(-K.sum(K.abs(vector[0] - vector[1]), axis=1, keepdims=True))


    malstm_distance = Lambda(exponent_neg_manhattan_distance, output_shape=out_shape)(
        [left_representation, right_representation])

    return Model([left_input, right_input], [malstm_distance])