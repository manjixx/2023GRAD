# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import pandas
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def data_loader():
    env = np.load('../dataset/summer/env.npy', allow_pickle=True).astype(float)         # ta hr
    season = np.load('../dataset/summer/season.npy', allow_pickle=True).astype(int)     # season
    date = np.load('../dataset/summer/date.npy', allow_pickle=True).astype(str)         # date
    body = np.load('../dataset/summer/body.npy', allow_pickle=True).astype(float)       # age height weight bmi griffith
    gender = np.load('../dataset/summer/gender.npy', allow_pickle=True).astype(int)     # gender
    label = np.load('../dataset/summer/label.npy', allow_pickle=True).astype(int)       # pmv

    # normalization
    x = np.concatenate((env, body), axis=1)
    x = scaler.fit_transform(x)

    # concatenate env body and gender
    # date season ta hr age height weight bmi griffith gender pmv
    x = np.concatenate((date, season, x, gender[:, None], label), axis=1)

    train_size = round(len(x) / 7 * 0.8)
    print(len(x)/7)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(0, train_size):
        start = i * 7
        end = (i + 1) * 7
        x_hat = x[start: end, :]
        y_hat = label[start: end, :]
        for j in range(0, 4):
            x_train.append(x_hat[j: j + 3, :])
            y_train.append(y_hat[j + 3: j + 4, :])

    for i in range(train_size, round(len(x) / 7)):
        start = i * 7
        end = (i + 1) * 7
        x_hat = x[start: end, :]
        y_hat = label[start: end, :]
        for j in range(0, 4):
            x_test.append(x_hat[j: j + 3, :])
            y_test.append(y_hat[j + 3: j + 4, :])

    print(f'train_feature shape: {(np.array(x_train).shape)}')
    print(f'test_feature shape: {(np.array(x_test).shape)}')

    return x_train, y_train, x_test, y_test


def R_loss(y_true, input):
    data = scaler.inverse_transform(input[:, 0:7])
    ta = data[:, 0:1]
    y_pred = input[:, 8:11]
    y_exp = []
    # ta 映射
    for i in range(0, len(ta)):
        if 28 >= ta[i] >= 26:
            y_exp.append(1)
        elif ta[i] < 26:
            y_exp.append(0)
        else:
            y_exp.append(2)
    y_exp = tf.one_hot(y_exp, depth=3)
    total = 0
    for i in range(0, len(y_pred)):
        p_true = tf.reshape(1 - y_exp[i], [1, 3])
        p_pred = tf.reshape(tf.math.log(alpha + y_pred[i]), [3, 1])
        r = tf.matmul(p_true, p_pred)
        total += r.numpy().item()
    r_loss = beta * total / len(y_pred)
    return r_loss


def CE_loss(y_true, y_pred):
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    ce_loss = tf.reduce_mean(loss)
    return ce_loss



if __name__ =='__main__':
    num_epochs = 195
    batch_size = 768
    learning_rate = 0.001
    scaler = MinMaxScaler()
    data_loader()

