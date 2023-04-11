# -*- coding: utf-8 -*-
"""
@Project ：data- mechanism
@File ：csv2npy_Ver1.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/5 20:18
"""

import pandas as pd
import numpy as np
import random
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if __name__ == '__main__':

    name = ['male', 'female',
            'young', 'old',
            'short', 'medium', 'tall',
            'thin', 'normal', 'fat',
            'bmi_l', 'bmi_n', 'bmi_h',
            'grf_l', 'grf_n', 'grf_h',
            'sen_l', 'sen_n', 'sen_h',
            'pre_l', 'pre_n', 'pre_h',
            'env_l', 'env_n', 'env_h',
            'date', 'time', 'ta', 'hr', 'season', 'va',
            'height_avg', 'weight_avg', 'bmi_avg', 'griffith_avg',
            'tsv']

    # 性别特征不需要归一化最后添加
    person_feature = ['male', 'female',
                      'young', 'old',
                      'short', 'medium', 'tall',
                      'thin', 'normal', 'fat',
                      'bmi_l', 'bmi_n', 'bmi_h',
                      'grf_l', 'grf_n', 'grf_h',
                      'sen_l', 'sen_n', 'sen_h',
                      'pre_l', 'pre_n', 'pre_h',
                      'env_l', 'env_n', 'env_h']
    env_feature = ['date', 'time', 'season', 'va', 'ta', 'hr']
    avg_feature = ['height_avg', 'weight_avg', 'bmi_avg']
    griffith = 'griffith_avg'
    count_feature = 'count'
    # 风速部分数据集需要自己生成
    y_feature = 'tsv'

    df = pd.read_csv('./dataset.csv').dropna(axis=0, how='any', inplace=False)

    # 房间人数
    count = df[count_feature].reset_index(drop=True)
    # 人员特征
    person = df[person_feature].reset_index(drop=True)
    # 环境数据
    env = df[env_feature].reset_index(drop=True)
    # 平均值
    avg = df[avg_feature].reset_index(drop=True)
    # 格里菲斯常数
    grf = df[griffith].reset_index(drop=True)
    # 标签
    tsv = df[y_feature].reset_index(drop=True)

    print(count, person, env, avg, grf, tsv)
    '''save data'''

    # np.save('./npy/count.npy', count)
    # np.save('./npy/person.npy', person)
    # np.save('./npy/env.npy', env)
    # np.save('./npy/avg.npy', avg)
    # np.save('./npy/grf.npy', grf)
    # np.save('./npy/tsv.npy', tsv)
