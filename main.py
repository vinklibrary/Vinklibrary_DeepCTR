# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: main.py
@time: 2019/5/23 15:13

这一行开始写关于本文件的说明与解释
"""

##########Package##########
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from utils.dataset import DeepffmDataset
from utils.data_preprocess import gen_features_sizes

from models import DeepFM
##########MainCode##########

# load data
train_data = DeepffmDataset(input_filepath="./DataSet/train_input_sample.csv",category_emb_filepath="./DataSet/category_emb_sample.csv",train=True)
loader_train = DataLoader(train_data, batch_size=256,
                          sampler=sampler.SubsetRandomSampler(range(9000)))
loader_val = DataLoader(train_data, batch_size=256,
                        sampler=sampler.SubsetRandomSampler(range(9000, 10000)))

features_sizes = gen_features_sizes('./DataSet/category_emb_sample.csv',39)

deepfm = DeepFM.DeepFM(field_size=39,feature_sizes=features_sizes,weight_decay=0.0001)
deepfm.fit(loader_train=loader_train,loader_val=loader_val,ealry_stopping=True)
