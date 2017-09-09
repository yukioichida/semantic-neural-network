from modules.log_config import LOG

from gensim.models.keyedvectors import KeyedVectors
from pathlib import Path

import os
import ntpath
import numpy as np

BASE_DIR = os.path.dirname(__file__)
PRE_EMBEDDING_MATRIX_DIR = os.path.join(Path(BASE_DIR).parent, 'embedding_matrix')

def load_embedding_matrix(dataset_name, embedding_file, word_index, embedding_dim = 300, binary = False):
    LOG.info('Loading embedding model from %s', embedding_file)
    _ , embedding_name = ntpath.split(embedding_file)
    embedding_matrix_file = os.path.join(PRE_EMBEDDING_MATRIX_DIR, dataset_name + '_' + embedding_name + '.npy')

    if os.path.exists(embedding_matrix_file):
        LOG.info('Loading existing embedding matrix')
        embedding_matrix = np.load(embedding_matrix_file)
    else:
        LOG.info('File %s not found. Loading new embedding matrix from: %s' % (embedding_matrix_file, embedding_name))
        embedding_model = KeyedVectors.load_word2vec_format(embedding_file, binary=binary)
        vocab_size = len(word_index) + 1
        embedding_matrix = 1 * np.random.randn(vocab_size, embedding_dim)  # This will be the embedding matrix
        embedding_matrix[0] = 0  # So that the padding will be ignored

        LOG.info('Creating the embedding matrix')
        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            if word in embedding_model.vocab:
                embedding_vector = embedding_model.word_vec(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector

        LOG.info('Embedding matrix as been created, removing embedding model from memory')
        del embedding_model
        LOG.info('Saving matrix in file %s' % (embedding_matrix_file))
        np.save(embedding_matrix_file, embedding_matrix)

    return embedding_matrix