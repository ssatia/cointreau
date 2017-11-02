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

gdax_client = gdax.PublicClient()

influxdb_client = InfluxDBClient(
    constants.INFLUXDB_HOST, constants.INFLUXDB_PORT,
    api_access_data.INFLUXDB_USER, api_access_data.INFLUXDB_PASS,
    constants.INFLUXDB_DB_NAME)

DATA_GRANULARITY = 60
HISTORICAL_DATA_BUFFER_SIZE = 10
SLEEP_TIME = 60
MODEL_DIR = 'model/'


def get_last_x_minute_data(currency, x):
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=x + HISTORICAL_DATA_BUFFER_SIZE)
    new_data = gdax_client.get_product_historic_rates(currency,
                                                      start_time.isoformat(),
                                                      end_time.isoformat(),
                                                      DATA_GRANULARITY)
    new_data = np.array(new_data)[-x:, 1:]

    return new_data


def get_last_minute_data(currency):
    new_data = get_last_x_minute_data(currency, 1)

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


def get_initial_state(currency, sequence_length):
    print("Collecting context for %s for the last %d mins." %
          (currency, sequence_length))

    price_series = get_last_x_minute_data(currency, sequence_length + 1)

    old_datapoint = price_series[0]
    for i in range(1, sequence_length + 1):
        new_datapoint = copy.copy(price_series[i])
        price_series[i] /= old_datapoint
        old_datapoint = new_datapoint
    price_series = price_series[1:, :]

    price_series = price_series.reshape(1, price_series.shape[0],
                                        price_series.shape[1])

    return price_series, old_datapoint


def write_prediction_to_influxdb(predicted_trend, actual_trend):
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
            constants.INFLUXDB_TAGS_TYPE: trend_type
        }
        datapoint[constants.INFLUXDB_FIELDS] = {
            constants.INFLUXDB_FIELDS_VALUE: trend_value
        }
        data.append(datapoint)

    influxdb_client.write_points(data)


def init(args):
    state, last_datapoint = get_initial_state(args.currency,
                                              args.sequence_length)

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
    pred = graph.get_tensor_by_name(trainer.OUTPUT_LAYER + ':0')

    db = MySQLdb.connect(
        db=constants.MYSQL_DB_NAME,
        host=constants.MYSQL_HOST,
        user=api_access_data.MYSQL_USER,
        passwd=api_access_data.MYSQL_PASSWD)
    db.autocommit(True)

    while (True):
        prediction = session.run([pred], {inputs: state})
        prediction = np.squeeze(prediction).item() - 1
        print('Trend prediction: %f%%' % (prediction * 100))

        if args.test:
            print('In test mode: not performing trades. Check grafana for '
                  'performance metrics')
        else:
            trade.trade(prediction, db)

        time.sleep(SLEEP_TIME)

        # Get new data
        new_datapoint = get_last_minute_data(args.currency)
        new_datapoint, last_datapoint = (new_datapoint / last_datapoint,
                                         new_datapoint)
        current_trend = new_datapoint[-2] - 1
        new_datapoint = new_datapoint.reshape(1, 1, new_datapoint.shape[0])
        state = np.concatenate((state[:, 1:, :], new_datapoint), axis=1)
        prediction *= 100
        current_trend *= 100
        write_prediction_to_influxdb(prediction, current_trend)
        print('Acutal trend: %f%%; Last prediction: %f%%' % (current_trend,
                                                             prediction))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trade using the pre-trained LSTM model')

    parser.add_argument(
        '-c',
        '--currency',
        default='ETH-USD',
        choices=set(('ETH-USD', 'BTC-USD')))
    parser.add_argument('-s', '--sequence_length', type=int, default=30)
    parser.add_argument('-m', '--model_file', default='')
    parser.add_argument('-t', '--test', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init(args)
