import api_access_data
import constants
import gdax
from influxdb import InfluxDBClient
import MySQLdb

import time

gdax_auth_client = gdax.AuthenticatedClient(api_access_data.GDAX_API_KEY,
                                            api_access_data.GDAX_API_SECRET,
                                            api_access_data.GDAX_PASSPHRASE)

influxdb_client = InfluxDBClient(
    constants.INFLUXDB_HOST, constants.INFLUXDB_PORT, constants.INFLUXDB_USER,
    constants.INFLUXDB_PASS, constants.INFLUXDB_DB_NAME)

CURRENCY = 'ETH-USD'
BUY_CONFIDENCE_THRESHOLD = 0.005
SELL_CONFIDENCE_THRESHOLD = 0
BASE_SIZE = 0.01
LIMIT_ORDER_BID_BUFFER = 0.01
LIMIT_ORDER_ASK_BUFFER = 0.01


def write_transaction_to_mysql(cursor, order):
    insert_query = """insert into %s values ('%s', %s, '%s', '%s', '%s', \
    '%s', '%s', '%s', %s, %s, %s)""" % (constants.TRANSACTIONS_TABLE,
                                        order[constants.GDAX_ID],
                                        order[constants.GDAX_SIZE],
                                        order[constants.GDAX_PRODUCT_ID],
                                        order[constants.GDAX_SIDE],
                                        order[constants.GDAX_TYPE],
                                        order[constants.GDAX_CREATED_AT],
                                        order[constants.GDAX_DONE_AT],
                                        order[constants.GDAX_DONE_REASON],
                                        order[constants.GDAX_FILL_FEES],
                                        order[constants.GDAX_FILLED_SIZE],
                                        order[constants.GDAX_EXECUTED_VALUE])
    cursor.execute(insert_query)

    # Update bankroll in MySQL
    transaction_cost = float(order[constants.GDAX_FILL_FEES]) + float(
        order[constants.GDAX_EXECUTED_VALUE])
    amount_filled = float(order[constants.GDAX_FILLED_SIZE])

    if (order[constants.GDAX_SIDE] == constants.ORDER_BUY):
        transaction_cost = -transaction_cost
        amount_filled = -amount_filled

    update_usd_query = """update %s set %s = %s + %f where %s='%s'""" % (
        constants.BANKROLL_TABLE, constants.BANKROLL_AMOUNT,
        constants.BANKROLL_AMOUNT, transaction_cost,
        constants.BANKROLL_CURRENCY, constants.BANKROLL_USD)
    cursor.execute(update_usd_query)

    update_eth_query = """update %s set %s = %s - %f where %s='%s'""" % (
        constants.BANKROLL_TABLE, constants.BANKROLL_AMOUNT,
        constants.BANKROLL_AMOUNT, amount_filled, constants.BANKROLL_CURRENCY,
        constants.BANKROLL_ETH)
    cursor.execute(update_eth_query)


def write_bankroll_to_influxdb(cursor):
    fetch_bankroll_query = """select %s, %s from %s""" % (
        constants.BANKROLL_CURRENCY, constants.BANKROLL_AMOUNT,
        constants.BANKROLL_TABLE)
    cursor.execute(fetch_bankroll_query)
    bankroll = cursor.fetchall()
    data = []
    for balance in bankroll:
        datapoint = {}
        datapoint[
            constants.
            INFLUXDB_MEASUREMENT] = constants.INFLUXDB_MEASUREMENT_BANKROLL
        datapoint[constants.INFLUXDB_TAGS] = {
            constants.INFLUXDB_TAGS_CURRENCY: balance[0]
        }
        datapoint[constants.INFLUXDB_FIELDS] = {
            constants.INFLUXDB_FIELDS_AMOUNT: balance[1]
        }
        data.append(datapoint)
    influxdb_client.write_points(data)


def handle_outstanding_orders(cursor):
    # Check on previous orders
    fetch_order_query = """select %s from %s""" % (
        constants.PENDING_ORDERS_ORDER_ID, constants.PENDING_ORDERS_TABLE)
    cursor.execute(fetch_order_query)
    pending_orders = cursor.fetchall()

    for order in pending_orders:
        # Cancel outstanding orders
        order_id = order[0]
        cancelled_order = gdax_auth_client.cancel_order(order_id)
        print(cancelled_order)

        order_status = gdax_auth_client.get_order(order_id)
        if (constants.GDAX_MESSAGE in order_status):
            print("Error: %s", order_status[constants.GDAX_MESSAGE])
        else:
            write_transaction_to_mysql(cursor, order_status)

    # Clear pending orders
    clear_orders_query = """delete from %s""" % (
        constants.PENDING_ORDERS_TABLE)
    cursor.execute(clear_orders_query)


def get_balance(currency, cursor):
    bal_query = """select %s from %s where %s='%s'""" % (
        constants.BANKROLL_AMOUNT, constants.BANKROLL_TABLE,
        constants.BANKROLL_CURRENCY, currency)
    cursor.execute(bal_query)

    return cursor.fetchone()[0]


def trade(pred, db):
    print(pred)
    cursor = db.cursor()

    handle_outstanding_orders(cursor)

    order_id = ''

    usd_bal = get_balance(constants.BANKROLL_USD, cursor)
    eth_bal = get_balance(constants.BANKROLL_ETH, cursor)
    ticker_data = gdax_auth_client.get_product_ticker(product_id=CURRENCY)

    # Check thresholds and place orders
    if (pred > BUY_CONFIDENCE_THRESHOLD):
        bid_price = ticker_data[constants.GDAX_BID]
        bid_price = float(bid_price) - LIMIT_ORDER_BID_BUFFER

        # TODO: Make this a function of the magnitude of the change and
        # available balance
        num_units = 1
        order_size = BASE_SIZE * num_units

        # Check to see if we have enough money
        if (usd_bal >= order_size * bid_price):
            result = gdax_auth_client.buy(
                price=bid_price,
                size=order_size,
                type=constants.ORDER_LIMIT,
                side=constants.ORDER_BUY,
                product_id=CURRENCY)
            print(result)

            if (constants.GDAX_ID in result):
                order_id = result[constants.GDAX_ID]

    elif (pred < SELL_CONFIDENCE_THRESHOLD):
        ask_price = ticker_data[constants.GDAX_ASK]
        ask_price = float(ask_price) + LIMIT_ORDER_ASK_BUFFER

        # TODO: Make this a function of the magnitude of the change and
        # available balance
        num_units = 1
        order_size = BASE_SIZE * num_units

        # Check to see if we have enough money
        if (eth_bal >= order_size):
            result = gdax_auth_client.sell(
                price=ask_price,
                size=order_size,
                type=constants.ORDER_LIMIT,
                side=constants.ORDER_SELL,
                product_id=CURRENCY)
            print(result)

            if (constants.GDAX_ID in result):
                order_id = result[constants.GDAX_ID]

    # Store new order in pending orders table
    if (order_id):
        store_order_query = """insert into %s values('%s')""" % (
            constants.PENDING_ORDERS_TABLE, order_id)
        cursor.execute(store_order_query)

    write_bankroll_to_influxdb(cursor)
