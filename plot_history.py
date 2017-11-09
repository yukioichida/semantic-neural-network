#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import pickle

from modules.configs import PRETRAIN, RESULTS_DIR


def plot_fit_history(history):
    # summarize history for loss
    axes = plt.gca()
    axes.set_ylim([0, 0.04])

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Custo')
    plt.xlabel('Iteração'.decode('utf-8'))
    plt.legend(['Treino', 'Validação'.decode('utf-8')], loc='upper right')
    plt.show()


if __name__ == '__main__':
    #history_file = 'results/model_1509895123609.history.p'
    model = 'model_1510006867660'
    history_file = 'results/{}.h5.history.p'.format(model)
    history = pickle.load(open(history_file, 'rb'))
    print history
    # history = pickle.loads(history)
    plot_fit_history(history)


