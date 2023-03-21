# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import pandas
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def data_loader():
    env = np.load('../dataset/summer/env.npy', allow_pickle=True).astype(float)  # ta hr va
    season = np.load('../dataset/summer/season.npy', allow_pickle=True).astype(int)  # season
    date = np.load('../dataset/summer/date.npy', allow_pickle=True)  # date
    body = np.load('../dataset/summer/body.npy', allow_pickle=True).astype(float)  # age height weight bmi griffith
    gender = np.load('../dataset/summer/gender.npy', allow_pickle=True).astype(int)  # gender
    label = np.load('../dataset/summer/label.npy', allow_pickle=True).astype(int)  # pmv

    # normalization
    # ta hr va age height weight bmi griffith
    x = np.concatenate((env, body), axis=1)
    x = scaler.fit_transform(x)
    # concatenate env body and gender
    # season ta hr  va age height weight bmi griffith gender pmv
    x = np.concatenate((season, x, gender[:, None], label), axis=1)

    print(len(x))
    train_size = round(len(x) / 7 * 0.8)
    print(len(x)/7)
    print(train_size)
    print(train_size * 7)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(0, train_size + 1):
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

    print(f'train_feature shape: {np.array(x_train).shape}')
    print(f'test_feature shape: {np.array(x_test).shape}')
    print(f'y_train shape: {np.array(y_train).shape}')
    print(f'y_test shape: {np.array(y_test).shape}')

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


class LSTMClassifier(tf.keras.Model):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.drop = tf.keras.layers.Dropout(rate=0.5)

        self.dense_M1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_M2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)

        self.dense_Tsk1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_Tsk2 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)

        self.dense_S1 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_S2 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)

        self.lstm = tf.keras.layers.LSTM(units=128, activation=tf.nn.leaky_relu, return_sequences=True)

        self.dense_PMV1 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu)
        self.dense_PMV2 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu)
        self.dense_PMV3 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu)
        self.dense_PMV4 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)
        self.dense_PMV5 = tf.keras.layers.Dense(units=3, activation=tf.nn.leaky_relu)

    def call(self, inputs, training=None, mask=None):
        # get data
        data = inputs['feature']  # season ta hr va age height weight bmi griffith gender pmv
        body = data[:, :, 4:10]
        env = data[:, :, 1:4]
        Ta = data[:, :, 1:2]
        Pa = tf.math.log1p(Ta)

        print(f'data shape: {data.shape}')
        print(f'Ta shape: {Ta.shape}')

        M_input = self.drop(body, training=training)
        M = self.dense_M1(M_input)
        M = self.drop(M, training=training)
        M = self.dense_M2(M)

        Tsk_input = self.drop(data, training=training)
        Tsk = tf.abs(self.dense_Tsk1(Tsk_input))
        Tsk_input = self.drop(Tsk, training=training)
        Tsk = tf.abs(self.dense_Tsk2(Tsk_input))
        Psk = tf.math.log1p(Tsk)

        # M, Tsk, Psk, Pa, S, season, ta, hr, va, age, height, weight, bmi, griffith, gender

        # print(f'M shape: {M.shape}')
        # print(f'Tsk shape: {Tsk.shape}')
        # print(f'Psk shape: {Psk.shape}')
        # print(f'Pa shape: {Pa.shape}')
        # print(f'env shape: {env.shape}')
        # print(f'body shape: {body.shape}')
        s_input = []
        for i in range(0, batch_size):
            s_input.append(tf.concat([M[i], Tsk[i], Psk[i], Pa[i], env[i], body[i]], axis=1))

        s_input = self.drop(s_input, training=training)
        S = self.dense_S1(s_input)
        s_input = self.drop(S, training=training)
        S = self.dense_S2(s_input)

        # M, Tsk, Psk, Pa, S, season ta hr  va age height weight bmi griffith gender pmv
        lstm_input = []
        for i in range(0, batch_size):
            lstm_input.append(tf.concat([M[i], Tsk[i], Psk[i], Pa[i], S[i], data[i]], axis=1))
        lstm_input = self.drop(lstm_input, training=training)

        lstm_input = self.drop(lstm_input, training=training)
        lstm = self.lstm(lstm_input)

        dense = self.dense_PMV1(lstm)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV2(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV3(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV4(dense)
        dense = self.drop(dense, training=training)
        dense = self.dense_PMV5(dense)

        output = tf.nn.softmax(dense)

        output = output[:, 2:3, :]

        # ta hr va age height weight bmi griffith
        # ta hr va age height weight bmi griffith gender pmv
        data = data[:, 2:3, 1:]

        x = []
        for i in range(0, batch_size):
            x.append(tf.concat((data[i], output[i]), axis=1))

        x = tf.reshape(x, [32, 1, 13])
        # print(f'output shape: {output.shape}')
        # print(f'x shape: {x.shape}')

        return [output, x]


def R_loss(y_true, input):
    input = tf.squeeze(input, axis=1)
    data = scaler.inverse_transform(input[:, 0:8])
    ta = data[:, 0:1]
    # ta hr va age height weight bmi griffith gender pmv y_pred
    y_pred = input[:, 10:]
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
    # print(f'y_pred shape: {y_pred.shape}')
    ce_sparse = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = ce_sparse(y_true, y_pred)
    ce_loss = tf.reduce_mean(loss)
    return ce_loss


def Accuracy(y_true, y_pred):
    y_pred = tf.squeeze(y_pred, axis=1)
    y_true = tf.reshape(y_true, [32, 1])
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_pred, y_true)


def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [Accuracy]
    loss = [CE_loss, R_loss]
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1,
                                                 mode='min', restore_best_weights=True)
    callbacks = [earlyStop]
    tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x={'feature': x_train},
              y=[y_train, y_train],
              epochs=num_epochs,
              batch_size=batch_size,
              validation_split=0.05,
              callbacks=callbacks,
              verbose=1,
              shuffle=True)
    checkpoint = tf.train.Checkpoint(classifier=model)
    path = checkpoint.save('save_model/model_lstm.ckpt')
    print("model saved to %s" % path)


def test():
    checkpoint = tf.train.Checkpoint(classifier=model)
    checkpoint.restore('save_model/model_lstm.ckpt-1').expect_partial()
    y_pred = model({'feature': x_test}, training=False)
    y_pred = tf.squeeze(y_pred[0], axis=1)
    y_test = tf.reshape(y_test, [32, 1])
    y_pred = np.argmax(y_pred, axis=1)
    print('准确率：' + str(accuracy_score(y_pred, y_test)))
    print('精确率 macro：' + str(precision_score(y_pred, y_test, average='macro')))
    print('精确率 micro：' + str(precision_score(y_pred, y_test, average='micro')))
    print('精确率 weighted：' + str(precision_score(y_pred, y_test, average='weighted')))
    print('Recall macro：' + str(recall_score(y_pred, y_test, average='macro')))
    print('Recall micro：' + str(recall_score(y_pred, y_test, average='micro')))
    print('Recall weighted：' + str(recall_score(y_pred, y_test, average='weighted')))
    print('F1-score macro：' + str(f1_score(y_pred, y_test, average='macro')))
    print('F1-score micro：' + str(f1_score(y_pred, y_test, average='micro')))
    print('F1-score weighted：' + str(f1_score(y_pred, y_test, average='weighted')))


if __name__ == '__main__':
    scaler = MinMaxScaler()

    num_epochs, batch_size, learning_rate = 128, 32, 0.008

    alpha, beta = 0, 0

    x_train, y_train, x_test, y_test = data_loader()

    for i in range(0, len(x_train)):
        print(i)
        print(x_train[i].shape)

    model = LSTMClassifier()
    train()
    test()
