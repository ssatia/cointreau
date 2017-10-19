from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import mean_squared_error

import math
import numpy as np

DATA_FILE_PATH = 'data/price_data_eth.csv'
MODEL_FILE_PATH = 'model/lstm_model_' + datetime.now().isoformat() + '.h5'
TRAIN_TEST_SPLIT = 0.8
LEARNING_RATE = 1e-2
NUM_NEURONS = 5
BATCH_SIZE = 1
NUM_EPOCHS = 10


def get_data():
    data = read_csv(DATA_FILE_PATH, header=None, index_col=0, usecols=[0, 4])
    # data.plot()
    # pyplot.show()
    data = data.values

    num_train = int(len(data) * TRAIN_TEST_SPLIT)

    train_x = data[:num_train]
    test_x = data[num_train:-1]

    data_offset = np.roll(data, -1)
    train_y = data_offset[:num_train]
    test_y = data_offset[num_train:-1]

    return (train_x, test_x, train_y, test_y)


def learn():
    train_x, test_x, train_y, test_y = get_data()

    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

    model = Sequential()
    model.add(
        LSTM(
            NUM_NEURONS,
            batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]),
            stateful=True))
    model.add(Dense(1))

    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    for i in range(NUM_EPOCHS):
        model.fit(
            train_x,
            train_y,
            epochs=1,
            batch_size=BATCH_SIZE,
            verbose=1,
            shuffle=False)
        model.reset_states()

    model.save(MODEL_FILE_PATH)
    print('Training completed. Model saved at:', MODEL_FILE_PATH)

    predictions = model.predict(test_x, batch_size=BATCH_SIZE)
    rmse = math.sqrt(mean_squared_error(test_y, predictions))

    print('Testing completed. RMSE:', rmse)


if __name__ == '__main__':
    learn()
