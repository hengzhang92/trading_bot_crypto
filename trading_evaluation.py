from binance.client import Client
from datetime import datetime
import pandas as pd
import numpy as np
import mykeys
from frequency_analysis import index_analyzer,buy_sell
import csv
import math

client = Client(mykeys.api_key, mykeys.api_secret)
coinlist=['BTC','ETH','XRP','BCH']

def profit_eval(coin):
    deposit=  client.get_deposit_history(asset=coin)
    orders = client.get_all_orders(symbol=coin + 'USDT', limit=1000)
    trades = client.get_my_trades(symbol=coin +'USDT')
    Trades = pd.DataFrame(trades)
    buy=sum(Trades['quoteQty'][Trades['isBuyer']==True].astype(float))
    sell=sum(Trades['quoteQty'][Trades['isBuyer']==False].astype(float))
    profit= math.nan
    if ~Trades['isBuyer'].iloc[-1]:
        if coin=='BTC':
            profit=sell-buy-1122
    return profit
profit_eval('BTC')

