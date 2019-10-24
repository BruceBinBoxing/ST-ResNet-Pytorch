# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torch.utils import data
import sys
sys.path.append('.')
from . import BikeNYC


T = 24
days_test = 10
len_test = T * days_test

class make_dataloader(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, mode, len_closeness, len_period,
                                   len_trend):
        'Initialization'
        self.dataset_name = dataset_name
        self.mode = mode
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend

        if self.dataset_name == 'bikenyc':
            print("loading data...")

            if self.mode == 'train':
                self.X_data, self.Y_data, _, _, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
                  len_closeness=self.len_closeness,
                  len_period=self.len_period,
                  len_trend=self.len_trend,
                  len_test=len_test,
                  preprocess_name='preprocessing.pkl',
                  meta_data=True)


            elif self.mode == 'test':
                _, _, self.X_data, self.Y_data, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
                    len_closeness=self.len_closeness,
                    len_period=self.len_period,
                    len_trend=self.len_trend,
                    len_test=len_test,
                    preprocess_name='preprocessing.pkl',
                    meta_data=True)

            assert len(self.X_data[0]) == len(self.Y_data)
            self.data_len = len(self.Y_data)

        else:
            print('Unknown datasets')

        self.mmn = mmn

    def __len__(self):
        'Denotes the total number of samples'
        return self.data_len

    def __str__(self):
        string = '' \
                 + '\tmode   = %s\n' % self.mode \
                 + '\tdataset name   = %s\n' % self.dataset_name \
                 + '\tmmn min   = %d\n' % self.mmn._min \
                 + '\tmmn max   = %d\n' % self.mmn._max \
                 + '\tlen    = %d\n' % len(self)

        return string

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X_c, X_p, X_t, X_meta = self.X_data[0][index], self.X_data[1][index], self.X_data[2][index] , self.X_data[3][index]
        y = self.Y_data[index]

        return X_c, X_p, X_t, X_meta, y

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    load(dataset_name='bikenyc')
