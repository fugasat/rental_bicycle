import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        print(logs)
        self.losses.append(logs.get('loss'))


def create_dataset(dataset, look_back):
    data_x, data_y = [], []
    for i in range(look_back, len(dataset)):
        data_x.append(dataset[i-look_back:i, :-1])
        data_y.append(dataset[i, -1:])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    return data_x, data_y


def create_sample_dataset(dataset, dataset_train, look_back):
    ds = dataset_train[-look_back:]
    ds = np.delete(ds, ds.shape[1] - 1, 1)
    ds = np.append(ds, dataset, axis=0)
    data_x = []
    for i in range(look_back, len(ds)):
        data_x.append(ds[i-look_back:i])
    data_x = np.array(data_x)
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])
    return data_x


def read_source_data(file_path):
    df = pd.read_table(file_path, engine='python')
    df = df.drop("id", axis=1)
    df = df.drop("dteday", axis=1)
    ds = df.values
    return ds


# 学習
ds_train = read_source_data('./train.tsv')
ds_sample = read_source_data('./test.tsv')

train_size = int(len(ds_train) * 0.67)
train, test = ds_train[0:train_size, :], ds_train[train_size:len(ds_train), :]

look_back = 7
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)
sample_x = create_sample_dataset(ds_sample, ds_train, look_back)

# LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(look_back, train_x.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

losshist = LossHistory()
"""
"""
model.fit(train_x, train_y,
          epochs=5, # 15
          batch_size=1,
          verbose=2,
          callbacks=[losshist])
with open('./model.pkl', mode='wb') as f:
    pickle.dump(model, f)

with open('./model.pkl', mode='rb') as f:
    model = pickle.load(f)

# 予測データの作成
sample_predict = model.predict(sample_x)

c = 8646
with open("./output.csv", mode='w') as f:
    for v in sample_predict:
        f.write("{0},{1}\n".format(c,int(v[0])))
        c += 1
