import argparse
from datetime import datetime
from matplotlib import pyplot
import math
import numpy as np
from pandas import read_csv
import tensorflow as tf

INPUT_PLACEHOLDER = 'input'
OUTPUT_LAYER = 'prediction'
TRAIN_TEST_SPLIT = 0.8
INPUT_SIZE = 2
OUTPUT_SIZE = 1


# Labelling for last timestep
def get_data(input_file, sequence_length):
    data = read_csv(input_file, header=None, usecols=[4, 5]).values

    # Make data stationary, based on % change
    prev = data[0][0]
    data[0][0] = 0
    for i in range(1, len(data)):
        data[i][0], prev = (data[i][0] - prev) / prev, data[i][0]

    data = np.array(data)
    x = []
    y = []

    for i in range(0, len(data) - sequence_length - 1, 1):
        x.append(data[i:i + sequence_length, ])
        y.append(data[i + sequence_length, 0])

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


def learn(args):
    train_x, test_x, train_y, test_y = get_data(args.input_file,
                                                args.sequence_length)

    inputs = tf.placeholder(
        tf.float32, (args.sequence_length, 1, INPUT_SIZE),
        name=INPUT_PLACEHOLDER)
    outputs = tf.placeholder(tf.float32, (None, OUTPUT_SIZE))

    num_hidden_units = args.hidden_units
    lstm_cell = tf.contrib.rnn.MultiRNNCell([
        tf.contrib.rnn.BasicLSTMCell(num_hidden_units),
        tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
    ])
    initial_state = lstm_cell.zero_state(1, tf.float32)
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, initial_state=initial_state, time_major=True)

    fc_weight = tf.Variable(tf.random_normal([num_hidden_units, OUTPUT_SIZE]))
    fc_bias = tf.Variable(tf.random_normal([OUTPUT_SIZE]))
    prediction = tf.add(
        tf.matmul(rnn_outputs[-1], fc_weight), fc_bias, name=OUTPUT_LAYER)

    error = tf.reduce_mean(tf.squared_difference(outputs, prediction))
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=args.learning_rate,
        decay=args.decay_rate).minimize(error)

    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for epoch in range(1, args.epochs + 1):
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

    save_path = saver.save(session, args.output_file)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an LSTM model to detect temporal patterns')

    parser.add_argument(
        '-i', '--input_file', default='data/price_data_eth.csv')
    parser.add_argument(
        '-o',
        '--output_file',
        default='model/lstm_model_' + datetime.now().isoformat() + '.ckpt')
    parser.add_argument('-n', '--hidden_units', type=int, default=100)
    parser.add_argument('-s', '--sequence_length', type=int, default=60)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-d', '--decay_rate', type=float, default=0.9)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    learn(args)
