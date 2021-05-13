from binance.client import Client
import pandas as pd
import numpy as np
import mykeys
import logging
import math
from scipy import signal
coins=['BTC','ETH','BNB','DOGE']
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

def get_last_price(coin,argument):
    # allow geting historical data for the last hours. with argument eg: '1000h ago UTC'
    klines = client.get_historical_klines( coin+'USDT', '15m', argument)
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


def buyorder_weighted(coin, totalasset, weight):
    # create buy orders based on total asset and weight.
    # eg: if totalasset=1000 USD and weight of BTC = 0.4 and existing BTC has a value = 200
    # function will create a buy order of 200 USD for BTC
    cash = float(client.get_asset_balance('USDT')['free'])
    avg_price = float(client.get_avg_price(symbol=coin + 'USDT')['price'])
    balance_USD = float(client.get_asset_balance(coin)['free'])*avg_price
    intended_USD_order = totalasset* weight-balance_USD
    if intended_USD_order>100: # doesn't make sense to create a buy order for 100USD
        if intended_USD_order> cash:
            USDQuantity = cash
        else:
            USDQuantity = intended_USD_order
        order = client.create_order(
        symbol=coin + 'USDT',
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_MARKET,
        quantity=round(USDQuantity / avg_price * (1 - 0.0015), 2))

def get_f(coin):
    temp=pd.read_csv('trading_frequency.csv',index_col='coins')
    buy_frequency = temp['buyf'][coin]
    sell_frequency = temp['sellf'][coin]
    invest_weight = temp['investweight'][coin]
    return buy_frequency,sell_frequency,invest_weight

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

def decision_buy_sell_f(sig,buy_f,sell_f,quantity,lastorder):
    # buy and sell order based on different filter frequency for bullish and bearish market
    numerator_coeffs, denominator_coeffs = signal.butter(2, buy_f)
    filtered_buy = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)
    numerator_coeffs, denominator_coeffs = signal.butter(2, sell_f)
    filtered_sell = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)
    if (filtered_sell[-2] > filtered_sell[-1]) and lastorder == 'BUY' and quantity>0.01:
        decision_out='sell'
    elif (filtered_buy[-2] < filtered_buy[-1]) and lastorder == 'SELL' and quantity<0.01:
        decision_out='buy'
    else:
        decision_out='NA'
    return decision_out

def get_total_asset(coins):
    # get total assets in USDT
    asset_total = float(client.get_asset_balance(asset='USDT')['free'])
    for coin in coins:
        sig = get_last_price(coin,'10h ago UTC')
        quantity = float(client.get_asset_balance(asset=coin)['free'])
        asset = sig.iloc[-1]*quantity
        asset_total += asset
    return asset_total


totalasset=get_total_asset(coins)
for coin in coins:
    sig=get_last_price(coin,'1000h ago UTC')
    buy_frequency,sell_frequency,invest_weight = get_f(coin)
    cash = float(client.get_asset_balance(asset='USDT')['free'])
    quantity = float(client.get_asset_balance(asset=coin)['free'])
    lastprice,lastorder=get_last_trade(coin)
    decision_out=decision_buy_sell_f(sig, buy_frequency, sell_frequency, quantity,lastorder)
    logger = setup_logger(coin, coin + '_decision.log')

    print(decision_out)
    try:
        if decision_out =='buy':
            buyorder_weighted(coin, totalasset, invest_weight-0.0001) # avoid numerical error
            logger.info('decision = ' + decision_out)
        elif decision_out == 'sell':
            sellorder(coin)
            logger.info('decision = ' + decision_out)
    except Exception as e:
        logger.error('issue with trading:'+ str(e))



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
