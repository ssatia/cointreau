import csv
import datetime
import gdax
import time

DATA_FILE_PATH = 'data/price_data'
CURRENCY_ETH = 'ETH-USD'
CURRENCY_BTC = 'BTC-USD'
DATA_GRANULARITY = 60


def collect_data():
    data_file_eth = open(DATA_FILE_PATH + '_eth.csv', 'a+')
    data_file_btc = open(DATA_FILE_PATH + '_btc.csv', 'a+')

    csv_writer_eth = csv.writer(data_file_eth)
    csv_writer_btc = csv.writer(data_file_btc)

    client = gdax.PublicClient()

    start_datetime = datetime.datetime(2017, 1, 1, 0, 0)
    end_datetime = datetime.datetime(2017, 11, 3, 0, 0)

    currencies = [CURRENCY_ETH, CURRENCY_BTC]
    csv_writers = [csv_writer_eth, csv_writer_btc]

    while (start_datetime < end_datetime):
        period_end_datetime = start_datetime + datetime.timedelta(hours=1)

        for (csv_writer_cur, currency) in zip(csv_writers, currencies):
            prices = client.get_product_historic_rates(
                currency,
                start_datetime.isoformat(),
                period_end_datetime.isoformat(), DATA_GRANULARITY)

            # Detect errors
            if (isinstance(prices, dict)):
                print('Error for %s (%s - %s):' %
                      (currency, start_datetime.isoformat(),
                       period_end_datetime.isoformat()))
                print(prices)
                continue

            prices.reverse()
            csv_writer_cur.writerows(prices)
            print('Wrote data for timestamp:', start_datetime, 'currency:',
                  currency)

        start_datetime = period_end_datetime

        # Prevent rate limiting
        time.sleep(1)

    print('Successfully completed writing historical data')


if __name__ == '__main__':
    collect_data()
