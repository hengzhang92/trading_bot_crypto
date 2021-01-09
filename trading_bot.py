from binance.client import Client
import pandas as pd
import numpy as np
import mykeys
import logging
import math
from scipy import signal
coins=['BTC','ETH','BNB']
client = Client(mykeys.api_key, mykeys.api_secret)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_last_price(coin):
    klines = client.get_historical_klines( coin+'USDT', '1h', '1000h ago UTC')
    col = ['open time', 'open', 'high', 'low', 'close', 'volume', 'colsetime', 'qoteAsetVolume', 'ntrade']
    Data = pd.DataFrame(klines)
    df = Data.iloc[:, :-3]
    df.columns = col
    sig=df['close'].astype(float)
    return sig

def sellorder(coin):
    balance= client.get_asset_balance(coin)
    order = client.create_order(
    symbol=coin+'USDT',
    side=Client.SIDE_SELL,
    type=Client.ORDER_TYPE_MARKET,
    quantity=math.floor(float(balance['free']) * 1000) / 1000)

def buyorder(coin):
    cash= float(client.get_asset_balance('USDT')['free'])
    avg_price = float(client.get_avg_price(symbol=coin+'USDT')['price'])
    order = client.create_order(
    symbol=coin+'USDT',
    side=Client.SIDE_BUY,
    type=Client.ORDER_TYPE_MARKET,
    quantity=round(cash/2/avg_price*(1-0.0015),2))

def get_dic():
    coins=pd.read_csv('trading_frequency.csv',index_col='coins')
    coins=coins['frequency']
    dic=coins.to_dict()
    return dic

def get_last_trade(coin):
    orders = client.get_all_orders(symbol=coin+'USDT', limit=1)
    last_price=float(orders[-1]['cummulativeQuoteQty'])/float(orders[-1]['executedQty'])
    return last_price, orders[-1]['side']



def decision(sig,f,quantity):
    numerator_coeffs, denominator_coeffs = signal.butter(2, f)
    filtered = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    if (filtered[-2] > filtered[-1]) and (filtered[-2] > filtered[-3]) and quantity>0.01:
        decision_out='sell'
    elif (filtered[-2] < filtered[-1]) and (filtered[-2] < filtered[-3]) and quantity<0.01:
        decision_out='buy'
    else:
        decision_out='NA'

    return decision_out

dic=get_dic()
for coin in coins:
    sig=get_last_price(coin)
    f=dic[coin]
    cash = float(client.get_asset_balance(asset='USDT')['free'])
    quantity = float(client.get_asset_balance(asset=coin)['free'])
    decision_out=decision(sig, f, quantity)
    logger = setup_logger(coin, coin + '_decision.log')

    print(decision_out)
    try:
        if decision_out =='buy':
            buyorder(coin)
            logger.info('decision = ' + decision_out)
        elif decision_out == 'sell':
            sellorder(coin)
            logger.info('decision = ' + decision_out)
    except:
        logger.error('issue with trading')



# balance = client.get_asset_balance(asset='BTC')
# klines=client.get_historical_klines('BTCUSDT','1h','1000h ago UTC')
# col=['open time','open','high','low','close','volume','colsetime','qoteAsetVolume','ntrade']
# Data=pd.DataFrame(test)
# df = Data.iloc[:,:-3]
# df.columns=col
# # order = client.create_order(
# # symbol='BTCUSDT',
# # side=Client.SIDE_SELL,
# # type=Client.ORDER_TYPE_MARKET,
# # quantity=balance['free'])
#
# Data=pd.DataFrame(klines)
# df = Data.iloc[:,:-3]
# filename=datetime.now().strftime("%Y%m%d%H%M%S")+'.pkl'
# col=['open time','open','high','low','close','volume','colsetime','qoteAsetVolume','ntrade']
# df.columns=col
# df.to_pickle(filename)
