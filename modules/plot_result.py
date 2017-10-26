#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os

from modules.configs import PRETRAIN, RESULTS_DIR


def plot_fit_history(history, model_file):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if PRETRAIN:
        plt.title('Com pré-treino'.decode('utf-8'))
    else:
        plt.title('Sem pré-treino'.decode('utf-8'))
    plt.ylabel('Custo')
    plt.xlabel('Iteração'.decode('utf-8'))
    plt.legend(['Treino', 'Validação'.decode('utf-8')], loc='upper left')
    filename = os.path.join(RESULTS_DIR, model_file + '.svg')
    plt.savefig(filename)
