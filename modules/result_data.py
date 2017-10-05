# -*- coding: utf-8 -*-
from modules.log_config import LOG
from modules.configs import *

import time
import os
import yaml
import scipy.stats as stats
import numpy as np
from sklearn.metrics import mean_squared_error as mse

RESULTS_DIR = 'results'


class ResultData:

    def __init__(self, pearson, spearman, mse, mae):
        self.pearson = pearson
        self.spearman = spearman
        self.mse = mse
        self.mae = mae

    def add_results(self, y_pred, y_val):
        samples_results = []
        for i in range(0, len(y_pred)):
            samples_results.append('%s - %s' % (y_pred[i], y_val[i]))
        self.results = samples_results

    def observation(self, obs):
        self.obs = obs

    def to_yaml(self):
        return {
            'input config': {
                'batch_size': BATCH_SIZE,
                'pretrain': PRETRAIN,
                'pretrain_epoch': PRETRAIN_EPOCHS,
                'train_epoch': TRAIN_EPOCHS,
                'embedding type': EMBEDDING_NAME,
                'dropout': DROPOUT,
                'recurrent_dropout': RECURRENT_DROPOUT,
                'stopwords': REMOVE_STOPWORDS
            },
            'pearson': self.pearson,
            'spearman': self.spearman,
            'mean squared error': self.mse,
            'mean absolute error': self.mae,
            'results': self.results,
            'obs': self.obs
        }

    def write(self):
        timestamp = int(round(time.time() * 1000))
        yaml_filename = 'bs_%s-pe_%s-e_%s-%s.yml' % (BATCH_SIZE,
                                                     PRETRAIN_EPOCHS,
                                                     TRAIN_EPOCHS,
                                                     timestamp)
        yaml_file = os.path.join(RESULTS_DIR, yaml_filename)

        with open(yaml_file, 'w') as file:
            yaml.dump(self.to_yaml(), file, default_flow_style=False)


def create_output(y_pred, y_test, mae, obs=''):
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

    result = ResultData(np.asscalar(pr_val), np.asscalar(sr_val), np.asscalar(mse_val), np.asscalar(mae))
    result.add_results(samples, gt)
    result.observation(obs)
    result.write()
