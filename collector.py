import csv
import datetime
import gdax
import time

DATA_FOLDER = 'data/'
PRODUCTS = ['BTC-USD', 'ETH-USD']
DATA_GRANULARITY = 60


def collect_data():
    csv_writers = []

    for product in PRODUCTS:
        data_file = open(DATA_FOLDER + product + '.csv', 'a+')
        csv_writers.append(csv.writer(data_file))

    client = gdax.PublicClient()

    start_datetime = datetime.datetime(2017, 1, 1, 0, 0)
    end_datetime = datetime.datetime(2017, 11, 3, 0, 0)

    while (start_datetime < end_datetime):
        period_end_datetime = start_datetime + datetime.timedelta(hours=1)

        for (csv_writer, product) in zip(csv_writers, PRODUCTS):
            prices = client.get_product_historic_rates(
                product,
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
            csv_writer.writerows(prices)
            print('Wrote data for timestamp %s for product %s' %
                  (start_datetime, product))

        start_datetime = period_end_datetime

        # Prevent rate limiting
        time.sleep(1)

    print('Successfully completed writing historical data')


if __name__ == '__main__':
    collect_data()
