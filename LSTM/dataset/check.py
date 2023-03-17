# -*- coding: utf-8 -*-
"""
@Project ：2023GRAD 
@File ：check.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/17 13:37 
"""

import pandas as pd

df = pd.read_csv(r'../../DataSet/2021.csv')

print(df.describe().transpose())