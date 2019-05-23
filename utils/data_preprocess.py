# encoding: utf-8
"""
@author: vinklibrary
@contact: shenxvdong1@gmail.com
@site: Todo

@version: 1.0
@license: None
@file: data_preprocess.py
@time: 2019/5/23 15:14

数据预处理模块，包括一下几个输出函数
1. 将文件文件转化为libffm格式文件
2.
"""
###########Package##########
import csv
import hashlib
import collections


###########OutFunction###########
def convert_csv2libffm(csv_file_path,libffm_file_path,features,frequent_feats):
    '''

    :param csv_file_path:
    :param libffm_file_path:
    :return: None
    '''
    with open(libffm_file_path, 'w') as f:
        for row in csv.DictReader(open(csv_file_path)):
            feats = [] # 初始化
            index = 1
            for feat in gen_feats(row,features):
                field = index
                index+=1
                if feat not in frequent_feats:
                    feat = feat.split('-')[0] + 'less'
                feats.append((field, feat))
            feats = gen_hashed_fm_feats(feats)
            f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')


def category_value_counts(csv_file_path, result_file_path, features):
    '''
    一般情况下，只统计训练集
    :param csv_file_path:
    :param result_file_path:
    :param category_size:
    :return:
    '''
    counts = collections.defaultdict(lambda: [0, 0, 0])

    for i, row in enumerate(csv.DictReader(open(csv_file_path)), start=1):
        label = row['Label']
        for field in features:
            value = row[field]
            if label == '0':
                counts[field + ',' + value][0] += 1
            else:
                counts[field + ',' + value][1] += 1
            counts[field + ',' + value][2] += 1

    with open(result_file_path, 'w') as f:
        f.write("Field,Value,Neg,Pos,Total,Ratio\n")
        for key, (neg, pos, total) in sorted(counts.items(), key=lambda x: x[1][2]):
            if total < 10:
                continue
            ratio = round(float(pos) / total, 5)
            f.write(key + ',' + str(neg) + ',' + str(pos) + ',' + str(total) + ',' + str(ratio) + '\n')

def gen_category_emb_from_libffmfile(filepath, dir_path):
    '''
    从libffm文件改为
    :param filepath:
    :param dir_path:
    :return:
    '''
    fr = open(filepath)
    cate_emb_arr = [{} for i in range(39)]
    for line in fr:
        datas = line.strip().split(' ')
        for item in datas[1:]:
            [filed, index, value] = item.split(':')
            filed = int(filed)
            index = int(index)
            if not index in cate_emb_arr[filed]:
                cate_emb_arr[filed][index] = len(cate_emb_arr[filed])

    with open(dir_path, 'w') as f:
        for i,item in enumerate(cate_emb_arr):
            for key in item:
                f.write(str(i)+','+str(key)+','+str(item[key])+'\n')

def gen_emb_input_file(libffm_filepath, category_emb_filepath, deepffm_input_filepath,features_size):
    cate_dict = load_category_index(category_emb_filepath,features_size)
    fr = open(libffm_filepath,'r')
    fw = open(deepffm_input_filepath,'w')
    for line in fr:
        row = []
        datas = line.strip().split(' ')
        row.append(datas[0])
        for item in datas[1:]:
            [filed, index, value] = item.split(':')
            filed = int(filed)
            row.append(str(cate_dict[filed][index]))
        fw.write(','.join(row)+'\n')

def gen_features_sizes(emb_file,features_size):
    result = []
    cate_dict = load_category_index(emb_file,features_size)
    for item in cate_dict:
        result.append(len(item))
    return result

###########InnerFunction###########
def gen_feats(row,features):
    '''

    :param row:
    :param numeric_size:
    :param category_size:
    :return:
    '''
    feats = []
    for field in features:
        value = row[field]
        key = field + '-' + value
        feats.append(key)
    return feats

def gen_hashed_fm_feats(feats,nr_bins=int(1e+6)):
    feats = ['{0}:{1}:1'.format(field - 1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (nr_bins - 1) + 1

def read_freqent_feats(value_counts_file_path,threshold=10):
    frequent_feats = set()
    for row in csv.DictReader(open(value_counts_file_path)):
        if int(row['Total']) < threshold:
            continue
        frequent_feats.add(row['Field']+'-'+row['Value'])
    return frequent_feats


def load_category_index(category_emb_filepath,features_size):
    f = open(category_emb_filepath,'r')
    cate_dict = []
    for i in range(features_size):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict

###########MainCode###########
if __name__ == '__main__':
    pass
