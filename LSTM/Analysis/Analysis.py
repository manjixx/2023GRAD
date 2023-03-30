# -*- coding: utf-8 -*-
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontManager


def font():
    mpl_fonts = set(f.name for f in FontManager().ttflist)
    print('all font list get from matplotlib.font_manager:')
    for f in sorted(mpl_fonts):
        print('\t' + f)


def split(data):
    hot_ta = data[(data[y_feature] == 2)][['ta']]
    hot_rh = data[(data[y_feature] == 2)][['hr']]
    cool_ta = data[(data[y_feature] == 0)][['ta']]
    cool_rh = data[(data[y_feature] == 0)][['hr']]
    com_ta = data[(data[y_feature] == 1)][['ta']]
    com_rh = data[(data[y_feature] == 1)][['hr']]
    return hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh


def distribution(title, data):
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)
    # 绘制分布图
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    label1 = axes.scatter(hot_ta, hot_rh, s=50, marker=None, c="red")
    label2 = axes.scatter(cool_ta, cool_rh, s=50, marker='x', c="blue")
    label3 = axes.scatter(com_ta, com_rh, s=50, marker='+', c="green")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title)
    plt.xlabel("temp(℃)")
    plt.ylabel("humid(%)")
    axes.legend((label1, label2, label3), ("hot", "cool", "comfort"), loc=3)
    plt.show()


def hist(season, data):
    hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh = split(data)
    # # 绘图风格
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)

    red = sns.color_palette("Set1")[0]

    # 绘制直方图
    sns.distplot(hot_ta, hist=True, hist_kws={'color': red}, kde_kws={'color': 'darkred', "shade": True, 'linestyle': '-'},
                 norm_hist=True)
    plt.xlabel(u"温度(℃)")
    # max = math.ceil(hot_ta.max())
    # min = math.floor(hot_ta.min())
    # x = np.arange(min, max, 1)
    # plt.xticks(x, rotation=45)
    plt.ylabel(u"数据比例")
    plt.title(u'2021年'+season+'热不适温度分布')
    plt.show()

    sns.distplot(com_ta, hist=True, hist_kws={'color': 'green'}, kde_kws={'color': 'darkgreen', "shade": True, 'linestyle': '-'},
                 norm_hist=True)
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
    plt.title(f'2021年{season}舒适温度分布')
    plt.show()

    sns.distplot(com_ta, hist=True, hist_kws={'color': 'blue'}, kde_kws={'color': 'darkblue', "shade": True, 'linestyle': '-'},
                 norm_hist=True)
    plt.xlabel(u"温度(℃)")
    plt.ylabel(u"数据比例")
    plt.title(f'2021年{season}冷不适温度分布')
    plt.show()
    # plt.hist(hot_rh, color='red')
    # plt.show()
    # plt.hist(com_rh, color='green')
    # plt.show()
    # plt.hist(cool_rh, color='blue')
    # plt.show()



if __name__ == '__main__':
    font()
    # 未经过序列化数据
    df = pd.read_csv('../../DataSet/2021.csv', encoding='gbk').dropna(axis=0, how='any', inplace=False)
    # print(df.shape[0])

    env = np.load('../dataset/2021/env.npy', allow_pickle=True).astype(float)  # ta hr va
    # print(env.shape)

    y_feature = 'thermal sensation'

    df = pd.read_csv('../../dataset/2021.csv').dropna(axis=0, how='any', inplace=False)
    # 标签数据
    df.loc[(df[y_feature] > 0.5), y_feature] = 2
    df.loc[((-0.5 <= df[y_feature]) & (df[y_feature] <= 0.5)), y_feature] = 1
    df.loc[(df[y_feature] < -0.5), y_feature] = 0

    df.loc[(df['time'] == '9:00:00'), 'time'] = '09:00:00'
    df.loc[(df['time'] == '9:30:00'), 'time'] = '09:30:00'
    data = df.drop(df.index[(df.time > '17:30:00') | (df.time >= '12:30:00') & (df.time <= '14:00:00')])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 20) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 56) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/20') & (data.no == 25) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/29') & (data.no == 33) & (data.room == 1)])
    data = data.drop(data.index[(data.date == '2021/7/25') & (data.no == 49) & (data.time == '12:00:00')])
    winter = data.loc[(data['season'] == 'winter')].reset_index(drop=True)
    summer = data.loc[(data['season'] == 'summer')].reset_index(drop=True)
    # print(winter.shape[0])
    # print(summer.shape[0])
    # print(summer)
    # print(winter)

    hist('夏季', summer)
    hist('冬季', winter)
    distribution('2021夏季数据分布图', summer)
    distribution('2021冬季数据分布图', winter)
    no = np.array(df['no'].unique())

