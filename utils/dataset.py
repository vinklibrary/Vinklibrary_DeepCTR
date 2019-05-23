# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: dataset.py
@time: 2019/5/23 16:53

这一行开始写关于本文件的说明与解释
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from utils.data_preprocess import load_category_index


class DeepffmDataset(Dataset):
    """
    Custom dataset class for Deepffm dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self,input_filepath,category_emb_filepath,train=True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - train: Train or test. Required.
        """
        self.train = train
        self.input_filepath = input_filepath
        self.category_emb_filepath = category_emb_filepath

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            result = {'label': [], 'index': [], 'value': [], 'feature_sizes': []}
            cate_dict = load_category_index(self.category_emb_filepath,39)
            for item in cate_dict:
                result['feature_sizes'].append(len(item))
            f = open(input_filepath, 'r')
            for line in f:
                datas = line.strip().split(',')
                result['label'].append(int(datas[0]))
                indexs = [int(item) for item in datas[1:]]
                values = [1 for i in range(len(indexs))]
                result['index'].append(indexs)
                result['value'].append(values)
            self.Xi = torch.from_numpy(np.array(result['index']).astype(np.float)).unsqueeze(-1)
            self.Xv = np.array(result['value'])
            self.target = np.array(result['label'])

        else:
            result = {'index': [], 'value': []}
            f = open(self.input_filepath, 'r')
            for line in f:
                datas = line.strip().split(',')
                indexs = [int(item) for item in datas[0:]]
                values = [1 for i in range(len(indexs))]
                result['index'].append(indexs)
                result['value'].append(values)
            self.Xi = result['index']
            self.Xv = result['value']

    def __getitem__(self, idx):
        if self.train:
            Xi = self.Xi[idx]
            Xv = self.Xv[idx]
            targetI = self.target[idx]
            return Xi, Xv, targetI
        else:
            Xi = self.Xi[idx]
            Xv = self.Xv[idx]
            return Xi, Xv

    def __len__(self):
        return len(self.Xi)


    def _check_exists(self):
        return os.path.exists(self.input_filepath) & os.path.exists(self.category_emb_filepath)
