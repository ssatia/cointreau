import argparse
import copy
from datetime import datetime
from matplotlib import pyplot
import math
import numpy as np
from pandas import read_csv
import tensorflow as tf

DATA_FOLDER = 'data/'
PRODUCTS = ['BTC-USD', 'ETH-USD', 'LTC-USD']
INPUT_PLACEHOLDER = 'input'
LABEL_PLACEHOLDER = 'label'
OUTPUT_LAYER = 'prediction'
TRAIN_TEST_SPLIT = 0.9
INPUT_SIZE = 5
OUTPUT_SIZE = 1
LOG_FREQUENCY = 100


def batchify(data, batch_size):
    batch = []
    for i in range(0, len(data) - batch_size, batch_size):
        batch.append(data[i:i + batch_size])

    return batch


def get_data(sequence_length, batch_size):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for (product_idx, product) in enumerate(PRODUCTS):
        data = read_csv(
            DATA_FOLDER + product + '.csv',
            header=None,
            usecols=[1, 2, 3, 4, 5]).values

        # Make data relative
        prev = data[0]
        for i in range(1, len(data)):
            cur_copy = copy.copy(data[i])
            data[i] = data[i] / prev
            prev = cur_copy
        data = data[1:, :]
        data = np.hstack((np.ones((len(data), 1)), data))
        data[:, 0] = product_idx

        x = []
        y = []
        for i in range(0, len(data) - sequence_length - 1, sequence_length):
            x.append(data[i:i + sequence_length, ])
            y.append([data[i + sequence_length, -2]])

        x, y = np.array(batchify(x, batch_size)), np.array(
            batchify(y, batch_size))
        num_train = int(len(x) * TRAIN_TEST_SPLIT)
        train_x.extend(x[:num_train])
        train_y.extend(y[:num_train])
        test_x.extend(x[num_train:])
        test_y.extend(y[num_train:])

    test_x, test_y = np.array(test_x), np.array(test_y)
    test_x = test_x.reshape(-1, test_x.shape[2], test_x.shape[3])
    test_y = test_y.reshape(-1, test_y.shape[2])

    return (train_x, test_x, train_y, test_y)


def learn(args):
    train_x, test_x, train_y, test_y = get_data(args.sequence_length,
                                                args.batch_size)

    inputs = tf.placeholder(
        tf.float32, (None, args.sequence_length, INPUT_SIZE),
        name=INPUT_PLACEHOLDER)
    outputs = tf.placeholder(tf.float32, (None, OUTPUT_SIZE))

    # Embedding for cryptocurency labels
    crypto_label = tf.placeholder(
        tf.int32, (None, args.sequence_length), name=LABEL_PLACEHOLDER)
    embedding_matrix = tf.Variable(
        tf.random_uniform([len(PRODUCTS), args.embedding_size], -1, 1))
    crypto_label_embeddings = tf.nn.embedding_lookup(embedding_matrix,
                                                     crypto_label)
    inputs_with_embeds = tf.concat([inputs, crypto_label_embeddings], axis=2)

    num_hidden_units = args.hidden_units

    if (args.layers > 1):
        lstm_cell = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
            for _ in range(args.layers)
        ])
    else:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden_units)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, output_keep_prob=args.dropout_prob)
    rnn_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    last_output = tf.gather(rnn_outputs, int(rnn_outputs.shape[0]) - 1)

    fc_weight = tf.Variable(
        tf.truncated_normal([num_hidden_units, OUTPUT_SIZE]))
    fc_bias = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]))
    prediction = tf.add(
        tf.matmul(last_output, fc_weight), fc_bias, name=OUTPUT_LAYER)

    error = tf.reduce_mean(tf.squared_difference(outputs, prediction))
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=args.learning_rate,
        decay=args.decay_rate).minimize(error)

    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    print('Beginning training.')
    for epoch in range(1, args.epochs + 1):
        epoch_error = 0
        for (counter, (x, y)) in enumerate(zip(train_x, train_y)):
            train_error, _, pred = session.run([error, optimizer, prediction],
                                               {
                                                   inputs: x[:, :, 1:],
                                                   crypto_label: x[:, :, 0],
                                                   outputs: y
                                               })
            if (counter % LOG_FREQUENCY == 0):
                print('Iteration: %d; Error: %f' % (counter, train_error))
            epoch_error += train_error

        epoch_error /= len(train_x)
        epoch_error = math.sqrt(epoch_error)
        print('Epoch #: %d; RMSE: %f' % (epoch, epoch_error))

    save_path = saver.save(session, args.output_file)
    print('Training completed. Model saved in file:', save_path)

    # Testing
    predictions, test_error = session.run([prediction, error], {
        inputs: test_x[:, :, 1:],
        crypto_label: test_x[:, :, 0],
        outputs: test_y
    })

    # Plot results
    predictions = predictions.flatten()
    test_y = test_y.flatten()
    timesteps = range(len(test_y))
    pyplot.plot(timesteps, test_y, label='Actual trend')
    pyplot.plot(timesteps, predictions, label='Predicted trend')
    pyplot.legend(loc='best')
    pyplot.show()

    test_error /= len(test_x)
    print('Test RMSE (in %):', math.sqrt(test_error))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an LSTM model to detect temporal patterns')

    parser.add_argument(
        '-o',
        '--output_file',
        default='model/lstm_model_' + datetime.now().isoformat() + '.ckpt')
    parser.add_argument('-n', '--hidden_units', type=int, default=100)
    parser.add_argument('-s', '--sequence_length', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-d', '--dropout_prob', type=float, default=0.8)
    parser.add_argument('--embedding_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--decay_rate', type=float, default=0.9)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    learn(args)
