import api_access_data
import argparse
import constants
import copy
from datetime import datetime, timedelta
from functools import reduce
import gdax
from influxdb import InfluxDBClient
import MySQLdb
import numpy as np
import tensorflow as tf
import time
import trade
import trainer

PRODUCTS = ['BTC-USD', 'ETH-USD', 'LTC-USD']
API_BACKOFF_TIME = 5
MODEL_DIR = 'model/'

gdax_client = gdax.PublicClient()

influxdb_client = InfluxDBClient(
    constants.INFLUXDB_HOST, constants.INFLUXDB_PORT,
    api_access_data.INFLUXDB_USER, api_access_data.INFLUXDB_PASS,
    constants.INFLUXDB_DB_NAME)


def get_last_x_interval_data(currency, x, interval_length):
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=x * interval_length)
    new_data = gdax_client.get_product_historic_rates(currency,
                                                      start_time.isoformat(),
                                                      end_time.isoformat(),
                                                      interval_length * 60)

    # Detect errors
    if (isinstance(new_data, dict)):
        print('Error for %s (%s - %s):' % (currency, start_time.isoformat(),
                                           end_time.isoformat()))
        print(new_data)

        # Try again
        time.sleep(API_BACKOFF_TIME)
        return get_last_x_interval_data(currency, x, interval_length)

    new_data.reverse()
    new_data = np.array(new_data)[-x:, 1:]

    return new_data


def get_last_interval_data(currency, interval_length):
    new_data = get_last_x_interval_data(currency, 1, interval_length)
    new_data = np.squeeze(np.array(merge_candles(new_data)))

    return new_data


def merge_candles(candles):
    merged_candle = []
    volume = reduce((lambda x, y: x + y[-1]), candles, 0)

    for i in range(len(candles[0]) - 1):
        weighted_avg = 0
        for candle in candles:
            weighted_avg += candle[i] * (candle[-1] / volume)
        merged_candle.append(weighted_avg)

    merged_candle.append(volume)

    return merged_candle


def get_initial_states(sequence_length, interval_length):
    states = []
    old_datapoints = []
    for (product_idx, product) in enumerate(PRODUCTS):
        print("Collecting context for %s for the last %d mins." %
              (product, interval_length * sequence_length))

        price_series = get_last_x_interval_data(product, sequence_length + 1,
                                                interval_length)

        old_datapoint = price_series[0]
        for i in range(1, sequence_length + 1):
            new_datapoint = copy.copy(price_series[i])
            price_series[i] = (price_series[i] / old_datapoint) - 1
            old_datapoint = new_datapoint
        price_series = price_series[1:, :]

        price_series = price_series.reshape(1, price_series.shape[0],
                                            price_series.shape[1])
        states.append(price_series)
        old_datapoints.append(old_datapoint)

    return states, old_datapoints


def write_prediction_to_influxdb(predicted_trend, actual_trend, product):
    data = []
    metrics = [(constants.INFLUXDB_TAGS_ACTUAL, float(actual_trend)),
               (constants.INFLUXDB_TAGS_PREDICTED,
                float(predicted_trend)), (constants.INFLUXDB_TAGS_ERROR, float(
                    abs(actual_trend - predicted_trend)))]

    for (trend_type, trend_value) in metrics:
        datapoint = {}
        datapoint[
            constants.
            INFLUXDB_MEASUREMENT] = constants.INFLUXDB_MEASUREMENT_PREDICTIONS
        datapoint[constants.INFLUXDB_TAGS] = {
            constants.INFLUXDB_TAGS_PRODUCT: product,
            constants.INFLUXDB_TAGS_TYPE: trend_type
        }
        datapoint[constants.INFLUXDB_FIELDS] = {
            constants.INFLUXDB_FIELDS_VALUE: trend_value
        }
        data.append(datapoint)

    influxdb_client.write_points(data)


def init(args):
    states, last_datapoints = get_initial_states(args.sequence_length,
                                                 args.interval_length)

    # Restore trained model
    session = tf.Session()
    ckpt_file = ''
    if not args.model_file:
        ckpt_file = tf.train.latest_checkpoint(MODEL_DIR)
    else:
        ckpt_file = args.model_file
    meta_graph = tf.train.import_meta_graph(ckpt_file + '.meta')
    meta_graph.restore(session, ckpt_file)
    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name(trainer.INPUT_PLACEHOLDER + ':0')
    labels = graph.get_tensor_by_name(trainer.LABEL_PLACEHOLDER + ':0')
    pred = graph.get_tensor_by_name(trainer.OUTPUT_LAYER + ':0')

    db = MySQLdb.connect(
        db=constants.MYSQL_DB_NAME,
        host=constants.MYSQL_HOST,
        user=api_access_data.MYSQL_USER,
        passwd=api_access_data.MYSQL_PASSWD)
    db.autocommit(True)

    while (True):
        predictions = []
        for (product_idx, (state, product)) in enumerate(
                zip(states, PRODUCTS)):
            crypto_labels = np.zeros((1, args.sequence_length))
            crypto_labels.fill(product_idx)
            prediction = session.run([pred], {
                inputs: state,
                labels: crypto_labels
            })
            prediction = np.squeeze(prediction).item() * 100
            predictions.append((prediction, product))
            print('Product: %s Trend prediction: %f%%' % (product, prediction))

        if args.test:
            print('In test mode: not performing trades. Check grafana for '
                  'performance metrics')
        else:
            # Process the predictions in order of magnitude of change
            predictions.sort(key=lambda x: abs(x[0]))
            for (prediction, product) in predictions:
                cursor = db.cursor()
                trade.trade(prediction, product, cursor)

        time.sleep(args.interval_length * 60)

        # Get new data
        for (idx, product) in enumerate(PRODUCTS):
            # Prevent rate limiting
            if (idx > 0):
                time.sleep(1)

            new_datapoint = get_last_x_interval_data(product, 1,
                                                     args.interval_length)
            new_datapoint = np.squeeze(new_datapoint)
            new_datapoint, last_datapoints[idx] = (
                (new_datapoint / last_datapoints[idx]) - 1, new_datapoint)
            current_trend = new_datapoint[-2]
            new_datapoint = new_datapoint.reshape(1, 1, new_datapoint.shape[0])
            states[idx] = np.concatenate(
                (states[idx][:, 1:, :], new_datapoint), axis=1)
            current_trend *= 100
            prediction = list(filter(lambda x: x[1] == product,
                                     predictions))[0][0]
            write_prediction_to_influxdb(prediction, current_trend, product)
            print('Product: %s Acutal trend: %f%%; Last prediction: %f%%' %
                  (product, current_trend, prediction))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trade using the pre-trained LSTM model')

    parser.add_argument('-s', '--sequence_length', type=int, default=10)
    parser.add_argument('-i', '--interval_length', type=int, default=15)
    parser.add_argument('-m', '--model_file', default='')
    parser.add_argument('-t', '--test', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init(args)
