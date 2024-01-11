import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('jena_climate_2009_2016.csv')
data.drop(data.iloc[:, 3:],axis = 1,inplace=True)
data.drop(columns='p (mbar)',inplace=True)

df = copy.copy(data)

df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
start_date = pd.to_datetime('2009-01-01')
end_date = pd.to_datetime('2012-12-31')
filtered_df = df[(df['Date Time'] >= start_date) & (df['Date Time'] <= end_date)]

filtered_df.set_index('Date Time', inplace=True)
daily_temperatures = filtered_df['T (degC)'].resample('H').mean()
daily_temperatures_df = daily_temperatures.to_frame(name='temperature')

daily_temperatures_df.reset_index(inplace=True)
daily_temperatures_df['Date Time'] = daily_temperatures_df['Date Time'].astype(str)

dataset = daily_temperatures_df.drop(columns=['index','Date Time'])
min_max_scaler = MinMaxScaler()
dataset = min_max_scaler.fit_transform(dataset)

latih, test, y_latih, y_test = train_test_split(dataset, dataset, test_size=0.2, shuffle=False) 

test = test.flatten()
latih = latih.flatten()

threshold_mae = (dataset.max() - dataset.min()) * 10/100

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

train_set = windowed_dataset(latih, window_size=72, batch_size=100, shuffle_buffer=1000)
val_set = windowed_dataset(test, window_size=72, batch_size=100, shuffle_buffer=1000)

model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64),
  tf.keras.layers.Dense(60, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
])

early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,epochs=100,validation_data = val_set,callbacks=[early_stopping])

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Plot mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Plot loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()