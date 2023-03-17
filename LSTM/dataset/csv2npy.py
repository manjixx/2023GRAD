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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def deal_2018():
    # 数据清洗
    df = pd.read_csv('../../dataset/2018.csv').dropna(axis=0, how='any', inplace=False)
    data = df[(df.time != '8:50:00') & (df.time != '14:20:00') & (df.time != '18:00:00')]
    data = data.drop(data.index[(data.no == 3) & (data.date == '2018/7/16')])
    data = data.drop(data.index[(data.no == 6) & (data.date == '2018/7/16')])

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label1 = data[y_feature].reset_index(drop=True).values.astype(int)
    # 人员特征

    body1 = data[body_feature].reset_index(drop=True)
    gender1 = data['gender'].reset_index(drop=True)

    # 环境特征
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    env1 = data[env_feature].reset_index(drop=True)
    env1 = pd.concat([env1, va], axis=1)

    return body1, gender1, env1, label1


if __name__ == '__main__':

    # 性别特征不需要归一化最后添加
    body_feature = ['age', 'height', 'weight', 'bmi', 'griffith']
    # 风速部分数据集需要自己生成
    env_feature = ['season', 'date', 'time', 'ta', 'hr']
    y_feature = 'thermal sensation'

    '''2018 summer'''
    body1, gender1, env1, label1 = deal_2018()


    ''' 2019 summer'''
    9 7-29


    df = pd.read_csv('../../dataset/2019_summer_clean.csv').dropna(axis=0, how='any', inplace=False)
    data = df[(df.time != '8:50:00') & (df.time != '14:20:00') & (df.time != '18:00:00')]
    data = data.drop(data.index[(data.no == 3) & (data.date == '2018/7/16')])
    data = data.drop(data.index[(data.no == 6) & (data.date == '2018/7/16')])

    # 标签数据
    data.loc[(data[y_feature] > 0.5), y_feature] = 2
    data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    data.loc[(data[y_feature] < -0.5), y_feature] = 0
    label1 = data[y_feature].reset_index(drop=True).values.astype(int)
    # 人员特征

    body1 = data[body_feature].reset_index(drop=True)
    gender1 = data['gender'].reset_index(drop=True)

    # 环境特征
    va = []
    for i in range(0, data.shape[0]):
        va.append(0.7 * round(random.random(), 1))
    va = pd.DataFrame({'va': va})
    env1 = data[env_feature].reset_index(drop=True)
    env1 = pd.concat([env1, va], axis=1)

    df = pd.read_csv('../../dataset/2019_summer.csv').dropna(axis=0, how='any', inplace=False)
    data = df[(df.time != '8:50:00') & (df.time != '14:20:00') & (df.time != '18:00:00')]
    # df_2021 = pd.read_csv('../../dataset/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    # data = df_2021.loc[df_2021['season'] == 'summer']

    ''' 2019 winter'''


    ''' 2021 '''


    # '''plot'''
    # hot_ta = data[(data['thermal sensation'] > 0.5)][['ta']]
    # hot_hr = data[(data['thermal sensation'] > 0.5)][['hr']]
    # cool_ta = data[(data['thermal sensation'] < -0.5)][['ta']]
    # cool_hr = data[(data['thermal sensation'] < -0.5)][['hr']]
    # com_ta = data[(data['thermal sensation'] <= 0.5) & (data['thermal sensation'] >= -0.5)][['ta']]
    # com_hr = data[(data['thermal sensation'] <= 0.5) & (data['thermal sensation'] >= -0.5)][['hr']]
    #
    # # 绘制分布图
    # plt.figure(figsize=(8, 5), dpi=80)
    # axes = plt.subplot(111)
    # label1 = axes.scatter(hot_ta, hot_hr, s=50, marker=None, c="red")
    # label2 = axes.scatter(cool_ta, cool_hr, s=50, marker='x', c="blue")
    # label3 = axes.scatter(com_ta, com_hr, s=50, marker='+', c="green")
    # plt.xlabel("temp(℃)")
    # plt.ylabel("humid(%)")
    # axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    # # plt.savefig('./result/pic/feedback distribution plot in ' + season + 'dataset.png')
    # plt.show()

    # '''environment data'''
    # va = []
    # for i in range(0, data.shape[0]):
    #     va.append(0.7 * round(random.random(), 1))
    # va = pd.DataFrame({'va': va})
    #
    # env_feature = ['ta', 'hr']
    # scaler = MinMaxScaler()
    # env = pd.DataFrame(scaler.fit_transform(data[env_feature]))
    # env_data = pd.concat([env, va], axis=1)
    #
    # '''body data'''
    # gender = pd.DataFrame(data['gender'].tolist())
    # body_feature = ['age', 'height', 'weight', 'bmi']
    # body_data = pd.DataFrame(MinMaxScaler().fit_transform(data[body_feature]))
    # body = pd.concat([gender, body_data], axis=1)
    #
    # '''label data'''
    #
    # y_feature = 'thermal sensation'
    # data.loc[(data[y_feature] > 0.5), y_feature] = 2
    # data.loc[((-0.5 <= data[y_feature]) & (data[y_feature] <= 0.5)), y_feature] = 1
    # data.loc[(data[y_feature] < -0.5), y_feature] = 0
    #
    # y = data[y_feature].values.astype(int)
    # print(y)
    # # y = data[y_feature]
    #
    # '''save data'''
    # np.save('../dataset/experimental_v1/env.npy', env_data)
    # np.save('../dataset/experimental_v1/body.npy', body)
    # np.save('../dataset/experimental_v1/label.npy', y)


