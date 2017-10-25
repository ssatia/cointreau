from datetime import datetime
from matplotlib import pyplot
import math
import numpy as np
from pandas import read_csv
import tensorflow as tf

DATA_FILE_PATH = 'data/price_data_eth.csv'
MODEL_FILE_PATH = 'model/lstm_model_' + datetime.now().isoformat() + '.ckpt'
INPUT_PLACEHOLDER = 'input'
OUTPUT_LAYER = 'prediction'
SEQUENCE_LENGTH = 60
TRAIN_TEST_SPLIT = 0.8
INPUT_SIZE = 2
NUM_HIDDEN_LSTM = 100
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-2
DECAY_RATE = 0.99
BATCH_SIZE = 1
NUM_EPOCHS = 1


# Labelling for last timestep
def get_data():
    data = read_csv(DATA_FILE_PATH, header=None, usecols=[4, 5]).values

    # Make data stationary, based on % change
    prev = data[0][0]
    data[0][0] = 0
    for i in range(1, len(data)):
        data[i][0], prev = (data[i][0] - prev) / prev, data[i][0]

    data = np.array(data)
    x = []
    y = []

    for i in range(0, len(data) - SEQUENCE_LENGTH - 1, 1):
        x.append(data[i:i + SEQUENCE_LENGTH, ])
        y.append(data[i + SEQUENCE_LENGTH, 0])

    x, y = np.array(x), np.array(y)
    num_train = int(len(x) * TRAIN_TEST_SPLIT)
    train_x = x[:num_train]
    test_x = x[num_train:]
    train_y = y[:num_train]
    test_y = y[num_train:]

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1,
                              train_x.shape[2])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1,
                            test_x.shape[2])
    train_y = train_y.reshape(train_y.shape[0], 1, 1)
    test_y = test_y.reshape(test_y.shape[0], 1, 1)

    return (train_x, test_x, train_y, test_y)


def learn():
    train_x, test_x, train_y, test_y = get_data()

    inputs = tf.placeholder(
        tf.float32, (SEQUENCE_LENGTH, BATCH_SIZE, INPUT_SIZE),
        name=INPUT_PLACEHOLDER)
    outputs = tf.placeholder(tf.float32, (None, OUTPUT_SIZE))

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN_LSTM)
    initial_state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, initial_state=initial_state, time_major=True)

    fc_weight = tf.Variable(tf.random_normal([NUM_HIDDEN_LSTM, OUTPUT_SIZE]))
    fc_bias = tf.Variable(tf.random_normal([OUTPUT_SIZE]))
    prediction = tf.add(
        tf.matmul(rnn_outputs[-1], fc_weight), fc_bias, name=OUTPUT_LAYER)

    error = tf.reduce_mean(tf.squared_difference(outputs, prediction))
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=LEARNING_RATE, decay=DECAY_RATE).minimize(error)

    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_error = 0
        for (x, y) in zip(train_x, train_y):
            train_error, _, pred = session.run([error, optimizer, prediction],
                                               {
                                                   inputs: x,
                                                   outputs: y
                                               })
            print(y, pred)
            epoch_error += train_error
            print('Train error:', train_error)

        epoch_error /= len(train_x)
        print('Epoch #:', epoch, '\nMSE:', epoch_error)

    save_path = saver.save(session, MODEL_FILE_PATH)
    print('Training completed. Model saved in file:', save_path)

    # Testing
    predictions = []
    test_error = 0
    max_error = 0
    for (x, y) in zip(test_x, test_y):
        pred, cur_error = session.run([prediction, error], {
            inputs: x,
            outputs: y
        })

        test_error += cur_error
        predictions.append(np.squeeze(pred))

    # Plot results
    test_y = list(map(lambda x: x * 100, test_y.flatten()))
    predictions = list(map(lambda x: x * 100, predictions))
    timesteps = range(len(test_y))
    pyplot.plot(timesteps, test_y, label='Actual trend')
    pyplot.plot(timesteps, predictions, label='Predicted trend')
    pyplot.legend(loc='best')
    pyplot.show()

    test_error /= len(test_x)
    print('Test RMSE (in %):', math.sqrt(test_error) * 100)


if __name__ == '__main__':
    learn()
