import glob
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
logging.basicConfig(filename=datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.log',level=logging.DEBUG)

def price_analysis(Euro, coin, sellprice, Data_raw, frequency, buy_tresh, sell_tresh, buysellratio, selection='open',
                   debug=0, actualoutput=0):
    trading_cost = 0.001
    Data_resampled = Data_raw.iloc[::frequency, :]
    Data_selected = Data_resampled[['open time', selection]].astype(float)
    Data = Data_selected[selection]
    buy_index = Data.pct_change() > buy_tresh
    sell_index = Data.pct_change() < sell_tresh
    tradingtimes = 0
    if sell_tresh > 0:
        sell_tresh = -sell_tresh
    actualbuy = []
    actualsell = []
    for index in Data.index:
        if buy_index[index] and (not Euro == 0) and Data[index] < sellprice * buysellratio:
            buyprice = Data[index]
            coin = Euro * (1 - trading_cost) / buyprice
            Euro = 0
            actualbuy = np.append(actualbuy, index)
            tradingtimes += 1
        if sell_index[index] and (not coin == 0):
            sellprice = Data[index]
            Euro = sellprice * coin * (1 - trading_cost)
            coin = 0
            tradingtimes += 1
            actualsell = np.append(actualsell, index)
    total_assets = Euro + coin * Data.iloc[-1]
    if debug:
        datetime.datetime.fromtimestamp(1347517370).strftime('%c')
        plt.plot(Data.index, Data)
        plt.plot(actualbuy, Data[actualbuy], '*b')
        plt.plot(actualsell, Data[actualsell], 'or')
    if actualoutput:
        return Euro, coin,sellprice,total_assets
    else:
        return total_assets

def randomlize_fixedsize(Data_raw,datasize,n=1):
    ctr = 0
    Data_selected = []
    while ctr < n:
        startingpoint = Data_raw.index[-1]
        while startingpoint > (Data_raw.index[-1] - datasize):
            startingpoint = random.choice(Data_raw.index)
        Data_selected.append(Data_raw.loc[startingpoint:startingpoint + datasize])
        ctr = ctr + 1
    return Data_selected

def find_tresh(Euro, coin, sellprice,Data,trading_frequencies,buy_treshholds,sell_treshholds,buy_selltresh,progress=False,getdf=False):
    assets = []
    output=[]
    i=0
    totalsize = trading_frequencies.size * buy_treshholds.size * sell_treshholds.size
    for frequency in trading_frequencies:
        for buy_treshhold in buy_treshholds:
            for sell_treshhold in sell_treshholds:
                asset = price_analysis(Euro, coin, sellprice,Data, frequency, buy_treshhold, sell_treshhold, buy_selltresh, selection='open', debug=0)
                assets.append([frequency, buy_treshhold, sell_treshhold, asset])
                output.append([frequency,buy_treshhold,sell_treshhold,asset])
                i += 1
                if progress:
                    print(i / totalsize)
    if getdf:
        df = pd.DataFrame(assets, columns=['frequency', 'buy_tresh', 'sell_tresh', 'totalasset'])
        return [output,df]
    else:
        return [output]

def determin_treshhold(Data,Euro, coin, sellprice,trading_frequencies,buy_treshholds,sell_treshholds,buy_selltresh):
    output=find_tresh(Euro, coin, sellprice,Data, trading_frequencies, buy_treshholds, sell_treshholds, buy_selltresh, progress=False, getdf=False)
    result=pd.DataFrame(output[0],columns=['trading_frequency', 'buy_tresh', 'sell_tresh', 'asset'])
    candidate=list(result[result.asset==result.asset.max()].values[0,0:3])
    msg = 'training result: parameter =' + str(candidate) + ' expected output=' + str(result.asset.max())
    logging.info(msg)
    return candidate


def fixedwidth_treshhold(Data,Euro, coin, sellprice,trading_frequencies,buy_treshholds,sell_treshholds,buy_selltresh,percentagewindowsize=0.25):
    # finding maximum gain treshhold with random selection of a fixed with
    # with percentage of the data size
    Data_randomlized = randomlize_fixedsize(Data,int(Data.shape[0]*percentagewindowsize) , 20)
    output = []
    for k, Data_loop in enumerate(Data_randomlized):
        print('training in process:' + str(k) + 'sample')
        output.append(
            find_tresh(Euro, coin, sellprice,Data_loop, trading_frequencies, buy_treshholds, sell_treshholds, buy_selltresh, progress=False, getdf=False))
    DF = []
    for i in output:
        DF.append(pd.DataFrame(i[0], columns=['trading_frequency', 'buy_tresh', 'sell_tresh', 'asset']))
    result = pd.concat(DF)
    T = result.pivot_table(index=['trading_frequency', 'buy_tresh', 'sell_tresh'], aggfunc='sum')
    candidate = T[T.asset == T.asset.max()].index.values[0]
    msg='training result: parameter =' + str(candidate) + ' expected output=' + str(T.asset.max()/10)
    logging.info(msg)
    return candidate
## This script is to test the short term inertia of financial product.
filename='./history/BTCUSDT.pkl'
Testing_windowsize=4*7*24 #1 month
Data=pd.read_pickle(filename)
Data=Data[0:10000]
startingindex=0
Euro=1
coin=0
sellprice=100000
buy_selltresh=2
pridction_perc=0.25
while startingindex< Data.shape[0]-Testing_windowsize:
    Data_cropped = Data[startingindex:startingindex + Testing_windowsize]
    #trading_param = fixedwidth_treshhold(Data_cropped, Euro, coin, sellprice, percentagewindowsize=pridction_perc)
    trading_frequencies = np.linspace(1, 9, 5).astype(np.int)
    buy_treshholds = np.linspace(0.005, 0.08, 10)
    sell_treshholds = np.linspace(-0.08, -0.005, 10)
    trading_param=determin_treshhold(Data_cropped, Euro, coin, sellprice, trading_frequencies, buy_treshholds, sell_treshholds,
                       buy_selltresh)


    Data_test = Data[startingindex + Testing_windowsize:int(startingindex + Testing_windowsize * 1.25)]
    Euro, coin,sellprice,total_assets = price_analysis(Euro, coin, sellprice, Data_test, int(trading_param[0]), trading_param[1], trading_param[2], buy_selltresh,
                          selection='open',
                          debug=0, actualoutput=1)
    msg='Testing result : Euro='+str(Euro) +' coin=' +str(coin)+' total_assets=' +str(total_assets)+ ' sellprice=' + str(sellprice)
    logging.info(msg)
    startingindex += int(pridction_perc*Testing_windowsize)




