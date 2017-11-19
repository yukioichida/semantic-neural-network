#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pickle


def plot_fit_history(history1, history2):
    # summarize history for loss
    axes = plt.gca()
    axes.set_ylim([0, 0.04])

    plt.plot(history1['loss'])
    plt.plot(history1['val_loss'])
    plt.plot(history2['loss'])
    plt.plot(history2['val_loss'])
    plt.ylabel('Custo')
    plt.xlabel('Iteração'.decode('utf-8'))
    plt.legend(['Treino LSTM', 'Validação LSTM'.decode('utf-8'), 'Treino GRU', 'Validação GRU'.decode('utf-8')], loc='upper right')
    plt.show()


if __name__ == '__main__':
    history_file_gru = 'results/model_1511069985328.h5.history.p'
    history_file_lstm = 'results/model_1509909192774.h5.history.p'
    #model = 'model_1510006867660'
    #history_file = 'results/{}.h5.history.p'.format(model)
    history_gru = pickle.load(open(history_file_gru, 'rb'))
    history_lstm = pickle.load(open(history_file_lstm, 'rb'))

    # history = pickle.loads(history)
    plot_fit_history(history_lstm, history_gru)


