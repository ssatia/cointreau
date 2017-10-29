import api_access_data
import argparse
import constants
from datetime import datetime, timedelta
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
    new_data = np.array(new_data)[-x:, 4:6]

    return new_data


def get_last_minute_data(currency):
    new_data = get_last_x_minute_data(currency, 2)

    return np.array(merge_candles(new_data))


def merge_candles(candles):
    volume = 0
    price = 0
    for candle in candles:
        volume += candle[1]
    for candle in candles:
        price += candle[0] * (candle[1] / volume)

    return [price, volume]


def get_initial_state(currency, sequence_length):
    print("Collecting context for %s for the last %d mins." %
          (currency, sequence_length))

    price_series = get_last_x_minute_data(currency, sequence_length + 1)

    stationary_data = []
    for i in range(1, sequence_length + 1):
        datapoint = [(
            price_series[i][0] - price_series[i - 1][0]) / price_series[i
                                                                        - 1][0]
                     ]
        datapoint.append(price_series[i][1])
        stationary_data.append(datapoint)

    stationary_data = np.array(stationary_data)
    stationary_data = stationary_data.reshape(stationary_data.shape[0], 1,
                                              stationary_data.shape[1])

    return stationary_data, price_series[-1][0]


def write_prediction_to_influxdb(predicted_trend, actual_trend):
    data = []
    for (trend_type, trend_value) in [(constants.INFLUXDB_TAGS_ACTUAL,
                                       float(actual_trend)),
                                      (constants.INFLUXDB_TAGS_PREDICTED,
                                       float(predicted_trend))]:
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
    state, last_price = get_initial_state(args.currency, args.sequence_length)

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
        prediction = np.squeeze(prediction).item()
        print('Trend prediction: %f%%' % (prediction * 100))

        if args.test:
            print('In test mode: not performing trades. Check grafana for '
                  'performance metrics')
        else:
            trade.trade(prediction, db)

        time.sleep(SLEEP_TIME)

        # Get new data
        new_data = get_last_minute_data(args.currency)
        new_price = new_data[0]
        new_data = new_data.reshape(1, 1, new_data.shape[0])
        state = np.vstack((state[1:, :], new_data))
        current_trend = (state[-1, 0, 0] - last_price) / last_price
        state[-1, 0, 0] = current_trend
        prediction *= 100
        current_trend *= 100
        write_prediction_to_influxdb(prediction, current_trend)
        print('Acutal trend: %f%%; Last prediction: %f%%' % (current_trend,
                                                             prediction))
        last_price = new_price


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trade using the pre-trained LSTM model')

    parser.add_argument(
        '-c',
        '--currency',
        default='ETH-USD',
        choices=set(('ETH-USD', 'BTC-USD')))
    parser.add_argument('-s', '--sequence_length', type=int, default=60)
    parser.add_argument('-m', '--model_file', default='')
    parser.add_argument('-t', '--test', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init(args)
