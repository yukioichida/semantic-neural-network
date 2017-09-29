# -*- coding: utf-8 -*-
from modules.log_config import LOG

import time
import os
import yaml
import scipy.stats as stats
import numpy as np
from sklearn.metrics import mean_squared_error as mse

RESULTS_DIR = 'results'

class InputConfiguration:

    def __init__(self, batch_size, pretrain_epoch, dropout, recurrent_dropout, epoch, embedding_type):
        self.batch_size = batch_size
        self.pretrain_epoch = pretrain_epoch
        self.epoch = epoch
        self.embedding_type = embedding_type
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout


class ResultData:

    def __init__(self, pearson, spearman, mse, mae, input_config):
        self.pearson = pearson
        self.spearman = spearman
        self.mse = mse
        self.mae = mae
        self.input_config = input_config

    def add_results(self, y_pred, y_val):
        samples_results = []
        for i in range(0, len(y_pred)):
            samples_results.append('%s - %s' % (y_pred[i], y_val[i]))
        self.results = samples_results

    def observation(self, obs):
        self.obs = obs

    def to_yaml(self):
        return {
            'input config' : {
                'batch_size' : self.input_config.batch_size,
                'pretrain_epoch' : self.input_config.pretrain_epoch,
                'train_epoch': self.input_config.epoch,
                'embedding type': self.input_config.embedding_type,
                'dropout': self.input_config.dropout,
                'recurrent_dropout': self.input_config.recurrent_dropout
            },
            'pearson' : self.pearson,
            'spearman' : self.spearman,
            'mean squared error': self.mse,
            'mean absolute error': self.mae,
            'results': self.results,
            'obs': self.obs
        }

    def write(self):
        timestamp = int(round(time.time() * 1000))
        yaml_filename = 'bs_%s-pe_%s-e_%s-%s.yml' % ( self.input_config.batch_size,
                                                    self.input_config.pretrain_epoch,
                                                    self.input_config.epoch,
                                                    timestamp )
        yaml_file = os.path.join(RESULTS_DIR, yaml_filename)

        with open(yaml_file, 'w') as file:
            yaml.dump(self.to_yaml(), file, default_flow_style=False)


def create_output(y_pred, y_test, mae, input_config, obs = '',scale=5):
    samples = y_pred.ravel()[:20]
    gt = y_test[:20]

    y_p = y_pred.ravel()
    y_t = y_test
    pr_val = stats.pearsonr(y_p, y_t)[0]
    sr_val = stats.spearmanr(y_p, y_t)[0]
    mse_val = mse(y_p, y_t)

    LOG.info(' Pearson: %f' % (pr_val))
    LOG.info(' Spearman: %f' % (sr_val))
    LOG.info(' MSE: %f' % (mse_val))

    result = ResultData(np.asscalar(pr_val), np.asscalar(sr_val), np.asscalar(mse_val), np.asscalar(mae), input_config)
    result.add_results(samples, gt)
    result.observation(obs)
    result.write()

