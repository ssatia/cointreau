from coinbase.wallet.client import Client
import api_access_data
import csv
import time


PRICE_AMOUNT = 'amount'
TIME_EPOCH = 'epoch'
TIME_ISO = 'iso'
DATA_FILE_PATH = 'data/price_data.csv'
LOG_FREQUENCY = 60


def get_client():
    return Client(api_access_data.API_KEY, api_access_data.API_SECRET)


def collect_data():
    data_file = open(DATA_FILE_PATH, 'a+')
    csv_writer = csv.writer(data_file)
    client = get_client()

    while(True):
        prices = []

        price = []
        timestamp = client.get_time()
        price.extend([timestamp[TIME_EPOCH], timestamp[TIME_ISO]])
        currency = 'BTC-USD'
        spot_price_btc = client.get_spot_price(currency_pair = currency)
        price.append(currency)
        price.append(spot_price_btc[PRICE_AMOUNT])
        prices.append(price)

        csv_writer.writerows(prices)
        print('Wrote data for timestamp:', timestamp[TIME_ISO], 'currency:', currency)

        time.sleep(60)


if __name__ == '__main__':
    collect_data()
