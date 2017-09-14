
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras import backend as K
import tensorflow as tf

# A entrada recebe os índices das palavras no vocabulário, para fazer o lookup na tabela de embeddings
max_sentence_length = 1
emb_dim = 3
left_input = Input(shape=(max_sentence_length,), dtype='int32')
right_input = Input(shape=(1,), dtype='int32')

embedding_matrix = 1 * np.random.randn(2, emb_dim)  # This will be the embedding matrix
embedding_matrix[0] = [1.1, 2.2, 3.3]
embedding_matrix[1] = [11.1, 22.2, 33.3]
emb = Embedding(2, emb_dim, weights=[embedding_matrix], input_length=max_sentence_length, trainable=False)

l = emb(left_input)
r = emb(right_input)
#r = Embedding(1, 1)(right_input)

def out_shape(shapes):
    return (shapes[0][0], 1)

def exponent_neg_manhattan_distance(vector):
    #    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    print(vector)
    return K.exp(-K.sum(K.abs(vector[0] - vector[1])))

def sum(vector):
    return K.sum(K.abs(vector[0]-vector[1]))
    #return K.sum(vector[0] + vector[1], axis=1, keepdims=True)

malstm_distance = Lambda(exponent_neg_manhattan_distance, output_shape=out_shape)([l, r])
#malstm_distance = Lambda(sum, output_shape=out_shape)([left_input, right_input])

#x1 = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
#x2 = np.array([[11.0, 22.0, 33.0], [1.0, 2.0, 3.0]])

x1 = np.array([[0]])
x2 = np.array([[1]])
model =  Model([left_input, right_input], [malstm_distance])

result = model.predict([x1, x2])
print(result)