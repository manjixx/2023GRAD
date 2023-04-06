# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：Analysis.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/4/5 23:12 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import matplotlib.ticker as ticker



def split(data):
    hot_ta = data[(data['bmi'] == 2)][['bmi']]
    hot_rh = data[(data['bmi'] == 2)][['bmi']]
    cool_ta = data[(data['bmi'] == 0)][['bmi']]
    cool_rh = data[(data['bmi'] == 0)][['bmi']]
    com_ta = data[(data['bmi'] == 1)][['bmi']]
    com_rh = data[(data['bmi'] == 1)][['bmi']]
    return hot_ta, hot_rh, com_ta, com_rh, cool_ta, cool_rh

def hist(xlabel, xticker, data):
    # # 绘图风格
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(style="white", palette='deep', font='Microsoft YaHei', font_scale=0.8)

    red = sns.color_palette("Set1")[0]
    bins = math.ceil(data.max()) - math.floor(data.min())
    weights = np.ones_like(np.array(data)) / float(len(data))
    # 绘制直方图
    sns.distplot(data,
                 bins=int(bins/xticker),
                 hist=True,
                 hist_kws={'color': red},
                 kde=False,
                 kde_kws={
                     'color': 'darkred',
                     "shade": True,
                     'linestyle': '--'
                 },
                 norm_hist=False)
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xticker))
    ax.tick_params("x",
                   which="major",
                   length=15,
                   width=1.0,
                   rotation=45)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel(xlabel)
    plt.ylabel(u"人数")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # font()
    df = pd.read_csv('../dataset/data/dataset.csv').dropna(axis=0, how='any', inplace=False)
    print(df)
    no = df['no'].unique()
    print(no)

    bmi = []
    griffith = []
    for n in no:
        data = df.loc[(df['no'] == n)]
        griffith.append((data['griffith'].unique()[0]))
        bmi.append((data['bmi'].unique()[0]))

    print(len(bmi))
    hist(u'BMI', 0.5, np.array(bmi))
    print(len(griffith))
    hist(u'热敏感度', 0.2, np.array(griffith))



