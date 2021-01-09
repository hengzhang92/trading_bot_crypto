from binance.client import Client
from datetime import datetime
import pandas as pd
import mykeys.py
client = Client(api_key, api_secret)
startdate='1 Dec, 2017'
enddate='30 Apr, 2020'
coinlist=['BTC','ETH','XRP','BCH','BNB','EOS','BSV','BCHABC','XTZ','XLM','LINK','ADA'
         'XMR','TRX','HT','DASH','ETC','NEO','ALGO','ATOM','IOTA','XEM','ONT',
         'CRO','DOGE','BAT','HBAR','VET','PAX','ZEC','MKR','FTT','EGT','DGB',
         'BTG','B2G','DCR','QTUM','LSK','ZRX','AMP','KNC','NRG','BTM','REP','RVN'
         'ICX','WAVES','BCD','ENJ','LEO','HYN','DAI','XIN','R','THETA','MONA','SC','MCO','NANO','DGD','KCS','OMG','SNT','STEEM','BEAM','NEXO','KMD','ZIL','BTS','NMR','XVG','DATA','HC'
          'MATIC','LEND','BCN','MANA','HOT','MAID','ZEN','IOST','GNT','XZC','STX','REN','COS','ELF','ARDR','CHZ','LRC','POWR','RCN','AE','ETN','PAI','BNK','FLC','STRAT'
          'NPXS','ANT','ORBS','WICC','AION','RLC','TOMO','WRX','GXS','ELA','RDD','BTT','CTXC','TCH','CNX','MCC','NULS','DIVI','TPAY','TNT','GNO','BAND','ARK','WTC','ABT']
stringin=[s + 'USDT' for s in coinlist]
col=['open time','open','high','low','close','volume','colsetime','qoteAsetVolume','ntrade']
errorlist=[]
for i ,s in enumerate(stringin):
    try:
        klines = client.get_historical_klines(s, Client.KLINE_INTERVAL_1HOUR, startdate, enddate)
        Data=pd.DataFrame(klines)
        df = Data.iloc[:,:-3]
        df.columns=col
        df.to_pickle('./history/' + s)
        print(i)
    except:
        errorlist.append(s)
