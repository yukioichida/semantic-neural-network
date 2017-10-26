# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
from modules.configs import PRETRAIN


def plot_fit_history(history, model_file):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if PRETRAIN:
        plt.title('Com pré-treino')
    else:
        plt.title('Sem pré-treino')
    plt.ylabel('Custo')
    plt.xlabel('Iteração')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    filename = os.path.join('figures', model_file + '.svg')
    plt.savefig(filename)
