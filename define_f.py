from binance.client import Client
from datetime import datetime
import pandas as pd
import numpy as np
import mykeys
from frequency_analysis import index_analyzer,buy_sell
import csv

client = Client(mykeys.api_key, mykeys.api_secret)
coinlist=['BTC','ETH','XRP','BCH']
col = ['open time', 'open', 'high', 'low', 'close', 'volume', 'colsetime', 'qoteAsetVolume', 'ntrade']
frequencies=np.linspace(0.01,0.1,20)
trading_f=[]
for coin in coinlist:
    coinstr=coin+'USDT'
    klines = client.get_historical_klines(coinstr, '1h', '20000h ago UTC')
    Data = pd.DataFrame(klines)
    df = Data.iloc[:, :-3]
    df.columns = col
    sig=df['open'].astype(float)
    assets=[]
    for f in frequencies:
        sell_index_A, buy_index_A = index_analyzer(sig, 1000, f)
        asset, actualbuy_index, actualsell_index = buy_sell(sig, buy_index_A, sell_index_A, 0.001, debug=False)
        assets.append(asset)
    frequency=frequencies[assets.index(max(assets))]

    trading_f.append(frequency)
d={'coins':coinlist,'frequency':trading_f}
result=pd.DataFrame(d)
result.to_csv('trading_frequency.csv')



