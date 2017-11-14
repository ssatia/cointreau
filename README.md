# cointreau

Cointreau is a cryptocurrency trading bot that collects market data, predicts price trends using an LSTM, and makes trades using the GDAX API.


## Components

`collector.py`: The collector collects historical market candle data using the GDAX API and stores the collected data in `data/`.

`trainer.py`: Trainer pulls the historical data and trains an LSTM model using a set of configurable parameters. Upon completion of training, the model is stored in `model\` (by default).

`trader.py`: The trader pulls an LSTM model (most recently trained model in `model\` by default), provides it with market data for the last _sequence_length_ minutes, and produces predictions for each cryptocurrency. It then calls `trade.py` in order of magnitude of predicted change for each product.

`trade.py`: Places a limit buy/sell order for the product if the prediction is above/below the _BUY_CONFIDENCE_THRESHOLD_/_SELL_CONFIDENCE_THRESHOLD_.


## Installation

`trade.py` uses an `api_access_data.py` file that stores the GDAX API key/secret, local MySQL DB user/pass and the local InfluxDB user/pass.

Create a file locally with the following format, fill in the relevant fields and save the file as `api_access_data.py`:
```
GDAX_PASSPHRASE = ''
GDAX_API_KEY = ''
GDAX_API_SECRET = ''

MYSQL_USER = ''
MYSQL_PASSWD = ''

INFLUXDB_USER = ''
INFLUXDB_PASS = ''
```

`trader.py` and `trade.py` also use a local MySQL DB and InfluxDB database named 'cointreau' with tables that store performance metrics, bankroll, pending and executed orders.

TODO: Add script to create these databases and tables automatically (assuming MySQL and InfluxDB are already installed locally).


## Results

TODO: Add model testing perf and Grafana trading performance metrics after I get the combined model working.
