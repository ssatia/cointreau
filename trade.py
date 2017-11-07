import api_access_data
import constants
from datetime import datetime
import gdax
from influxdb import InfluxDBClient
import MySQLdb

BUY_CONFIDENCE_THRESHOLD = 0.02
SELL_CONFIDENCE_THRESHOLD = -0.01
BASE_SIZE = 0.01
LIMIT_ORDER_BID_BUFFER = 0.01
LIMIT_ORDER_ASK_BUFFER = 0.01

gdax_auth_client = gdax.AuthenticatedClient(api_access_data.GDAX_API_KEY,
                                            api_access_data.GDAX_API_SECRET,
                                            api_access_data.GDAX_PASSPHRASE)

influxdb_client = InfluxDBClient(
    constants.INFLUXDB_HOST, constants.INFLUXDB_PORT,
    api_access_data.INFLUXDB_USER, api_access_data.INFLUXDB_PASS,
    constants.INFLUXDB_DB_NAME)


def write_transaction_to_mysql(cursor, order, fiat_currency, crypto_currency):
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

    update_fiat_query = """update %s set %s = %s + %f where %s='%s'""" % (
        constants.BANKROLL_TABLE, constants.BANKROLL_AMOUNT,
        constants.BANKROLL_AMOUNT, transaction_cost,
        constants.BANKROLL_CURRENCY, fiat_currency)
    cursor.execute(update_fiat_query)

    update_crypto_query = """update %s set %s = %s - %f where %s='%s'""" % (
        constants.BANKROLL_TABLE, constants.BANKROLL_AMOUNT,
        constants.BANKROLL_AMOUNT, amount_filled, constants.BANKROLL_CURRENCY,
        crypto_currency)
    cursor.execute(update_crypto_query)


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


def handle_outstanding_orders(cursor, fiat_currency, crypto_currency):
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
            print("Error: %s" % (order_status[constants.GDAX_MESSAGE]))
        else:
            write_transaction_to_mysql(cursor, order_status, fiat_currency,
                                       crypto_currency)

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


def trade(pred, product, cursor):
    print('Trading with trend prediction of %f for %s at %s' %
          (pred, product, datetime.now()))

    crypto_currency, fiat_currency = product.split('-')
    handle_outstanding_orders(cursor, fiat_currency, crypto_currency)

    crypto_bal = get_balance(crypto_currency, cursor)
    fiat_bal = get_balance(fiat_currency, cursor)
    ticker_data = gdax_auth_client.get_product_ticker(product_id=product)
    order_id = ''

    # Check thresholds and place orders
    if (pred > BUY_CONFIDENCE_THRESHOLD):
        bid_price = ticker_data[constants.GDAX_BID]
        bid_price = float(bid_price) - LIMIT_ORDER_BID_BUFFER

        # TODO: Make this a function of the magnitude of the change and
        # available balance
        num_units = 1
        order_size = BASE_SIZE * num_units

        # Check to see if we have enough money
        if (fiat_bal >= order_size * bid_price):
            result = gdax_auth_client.buy(
                price=bid_price,
                size=order_size,
                type=constants.ORDER_LIMIT,
                side=constants.ORDER_BUY,
                product_id=product)
            print('Place limit buy order at %f for %f unit(s) of %s on %s' %
                  (bid_price, order_size, crypto_currency, product))
            print(result)

            if (constants.GDAX_ID in result):
                order_id = result[constants.GDAX_ID]
        else:
            print('%s balance too low to purchase unit(s) on %s' %
                  (fiat_currency, product))

    elif (pred < SELL_CONFIDENCE_THRESHOLD):
        ask_price = ticker_data[constants.GDAX_ASK]
        ask_price = float(ask_price) + LIMIT_ORDER_ASK_BUFFER

        # TODO: Make this a function of the magnitude of the change and
        # available balance
        num_units = 1
        order_size = BASE_SIZE * num_units

        # Check to see if we have enough money
        if (crypto_bal >= order_size):
            result = gdax_auth_client.sell(
                price=ask_price,
                size=order_size,
                type=constants.ORDER_LIMIT,
                side=constants.ORDER_SELL,
                product_id=product)
            print('Place limit sell order at %f for %f unit(s) of %s on %s' %
                  (ask_price, order_size, crypto_currency, product))
            print(result)

            if (constants.GDAX_ID in result):
                order_id = result[constants.GDAX_ID]
        else:
            print('No units of %s available to sell on %s' % (crypto_currency,
                                                              product))

    else:
        print('Predicted trend of %f for %s is below the buy confidence \
            threshold and above the sell confidence threshold.' % (pred,
                                                                   product))

    # Store new order in pending orders table
    if (order_id):
        store_order_query = """insert into %s values('%s')""" % (
            constants.PENDING_ORDERS_TABLE, order_id)
        cursor.execute(store_order_query)

    write_bankroll_to_influxdb(cursor)
