import glob
import pandas as pd
import matplotlib
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
def price_analysis(Euro,coin,sellprice,Data_raw,frequency,buy_tresh,sell_tresh,buysellratio,selection='open',debug=0,actualoutput=0):
    trading_cost = 0.001
    Data_resampled = Data_raw.iloc[::frequency, :]
    Data_selected = Data_resampled[['open time', selection]].astype(float)
    Data=Data_selected[selection]
    buy_index = Data.pct_change() > buy_tresh
    sell_index = Data.pct_change() < sell_tresh
    tradingtimes = 0
    if sell_tresh>0:
        sell_tresh=-sell_tresh
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
        plt.plot(Data.index,Data)
        plt.plot(actualbuy,Data[actualbuy],'*b')
        plt.plot(actualsell,Data[actualsell],'or')
    if actualoutput:
        return Euro,coin
    else:
        return total_assets


def randomlize(Data_raw,n=1,minimumsize=168):
    # select n number of radomly chosen historical data
    data_length=Data_raw.shape[0]
    ctr=0
    Data_selected=[]
    while ctr< n:
        startingpoint = data_length
        while startingpoint>(Data_raw.index[-1]-minimumsize):
            startingpoint = random.choice(Data_raw.index)
        endpoint=random.choice(np.arange(startingpoint+minimumsize,data_length))
        Data_selected.append(Data_raw[startingpoint:endpoint])
        ctr=ctr+1
    return Data_selected

def randomlize_fixedsize(Data_raw,datasize,n=1):
    data_length = Data_raw.shape[0]
    ctr = 0
    Data_selected = []
    while ctr < n:
        startingpoint = data_length
        while startingpoint > (Data_raw.index[-1] - datasize):
            startingpoint = random.choice(Data_raw.index)
        Data_selected.append(Data_raw[startingpoint:startingpoint + datasize])
        ctr = ctr + 1
    return Data_selected



def find_tresh(Data,trading_frequencies,buy_treshholds,sell_treshholds,buy_selltresh,progress=False,getdf=False):
    assets = []
    output=[]
    i=0
    totalsize = trading_frequencies.size * buy_treshholds.size * sell_treshholds.size
    for frequency in trading_frequencies:
        for buy_treshhold in buy_treshholds:
            for sell_treshhold in sell_treshholds:
                asset = price_analysis(Data, frequency, buy_treshhold, sell_treshhold, buy_selltresh, selection='open', debug=0)
                assets.append([frequency, buy_treshhold, sell_treshhold, asset])
                if asset>1:
                    output.append([frequency,buy_treshhold,sell_treshhold,asset])
                i += 1
                if progress:
                    print(i / totalsize)
    if getdf:
        df = pd.DataFrame(assets, columns=['frequency', 'buy_tresh', 'sell_tresh', 'totalasset'])
        return [output,df]
    else:
        return [output]

def find_maxgain_tresh(Data,trading_frequencies,buy_treshholds,sell_treshholds,buy_selltresh,progress=False):
    maxasset = 0
    i = 0
    totalsize = trading_frequencies.size * buy_treshholds.size * sell_treshholds.size
    for frequency in trading_frequencies:
        for buy_treshhold in buy_treshholds:
            for sell_treshhold in sell_treshholds:
                asset = price_analysis(Data, frequency, buy_treshhold, sell_treshhold, buy_selltresh, selection='open', debug=0)
                if asset>maxasset:
                    maxasset = asset
                    parameter=[frequency,buy_treshhold,sell_treshhold]

                if progress:
                    print(i / totalsize)
                    i=i+1
    return parameter






def determin_tresh(Data,trainingsize,testingsize,debug_fig):



    Data_randomlized = randomlize(Data, trainingsize)
    trading_frequencies = np.arange(1, 25)
    buy_treshholds = np.arange(0.004, 0.08, 0.002)
    sell_treshholds = np.arange(-0.08, -0.004, 0.002)
    output=[]
    for k,Data_loop in enumerate(Data_randomlized):
        print('training in process:' + str(k) + 'sample')
        output.append(find_tresh(Data_loop,trading_frequencies,buy_treshholds,sell_treshholds,progress=False,getdf=False))
    DF=[]
    for i in output:

        DF.append(pd.DataFrame(i[0],columns=['trading_frequency','buy_tresh','sell_tresh']))
    result=pd.concat(DF)
    T = result.pivot_table(index=['trading_frequency', 'buy_tresh', 'sell_tresh'], aggfunc='size')
    candidates=T[T==T.max()]
    #test 10 random samples
    Data_test=randomlize(Data, n=testingsize)
    matrix_out=[]
    for k,candidate in enumerate(candidates.index):
        print('finding tresh:' + str(k/len(candidates)) + 'candidate')
        frequency=candidate[0]
        buy_tresh=candidate[1]
        sell_tresh=candidate[2]
        out=[]
        for sample in Data_test:
            out.append(price_analysis(sample, frequency, buy_tresh, sell_tresh, 1, selection='open', debug=0))
    matrix_out.append(out)
    DF = pd.DataFrame(matrix_out)
    Trading_param=candidates.index[DF.mean(axis=1).idxmax()]
    average_return=DF.mean(axis=1).max()
    confidence=sum(DF.iloc[DF.mean(axis=1).idxmax()]>1)/len(DF.columns)



    if debug_fig:
        fig, ax = plt.subplots(3, 3, sharex=True)
        labels=['open','high','low']
        for i in np.arange(ax.shape[0]):
            for j in np.arange(ax.shape[1]):
                plt.axes(ax[i, j])
                Data_test = randomlize(Data, n=1, minimumsize=168)
                if i ==0:
                    test_out = price_analysis(Data_test[0], int(Trading_param[0]), Trading_param[1], Trading_param[2],
                                              1,
                                              selection='open', debug=1)
                if i==1:
                    test_out = price_analysis(Data_test[0], int(Trading_param[0]), Trading_param[1], Trading_param[2],
                                              1,
                                              selection='high', debug=1)
                if i==2:
                    test_out = price_analysis(Data_test[0], int(Trading_param[0]), Trading_param[1], Trading_param[2],
                                              1,
                                              selection='low', debug=1)



                ax[i, j].title.set_text(labels[i] + str(test_out.round(3)))
    return Trading_param,average_return,confidence,fig

if __name__ == "__main__":
    files = glob.glob('./history/*.pkl')
    for file in files[0:5]:
        Data = pd.read_pickle(file)
        trainingsize=24*7*4
        Data_training =Data[-trainingsize*2:-trainingsize]
        Data_training.reset_index(inplace=True)
        Data_testing= Data[-trainingsize:]
        Data_testing.reset_index(inplace=True)

        Trading_param,average_return,confidence,fig=determin_tresh(Data_training, 5, 10, 1)
        fig.savefig('./dataresult/figures/' + file.strip('./history/.pkl') + '.jpg')
        fig_test=plt.figure()
        assets_test = price_analysis(Data_testing, Trading_param[0], Trading_param[1], Trading_param[2], 1, selection='open',
                                     debug=1)
        fig_test.savefig('./dataresult/figures/' + file.strip('./history/.pkl') + '_test.jpg')

        plt.close()
        saved_param=np.append(Trading_param,[average_return,confidence,assets_test])
        np.savetxt('./dataresult/' + file.strip('./history/.pkl') + '.csv', saved_param)


    plot=Falsetesting in process
    if plot:
        fig, ax = plt.subplots(3, 3, sharex=True)
        labels = ['open', 'high', 'low']
        for i in np.arange(ax.shape[0]):
            for j in np.arange(ax.shape[1]):
                plt.axes(ax[i, j])
                Data_test = randomlize(Data, n=1, minimumsize=168)
                if i == 0:
                    test_out = price_analysis(Data_test[0], int(Trading_param[0]), Trading_param[1], Trading_param[2],
                                              1,
                                              selection='low', debug=1)
                if i == 1:
                    test_out = price_analysis(Data_test[0], int(Trading_param[0]), Trading_param[1], Trading_param[2],
                                              1,
                                              selection='low', debug=1)
                if i == 2:
                    test_out = price_analysis(Data_test[0], int(Trading_param[0]), Trading_param[1], Trading_param[2],
                                              1,
                                              selection='low', debug=1)
                ax[i, j].title.set_text(labels[i] + str(test_out.round(3)))
        df=df.astype(float)
        df['gainpercent']=(df.totalasset-1)*100
        # plot maximum
        df_max=df[df['gainpercent']==df['gainpercent'].max()]
        plt.figure()
        selection=0
        result = price_analysis(Data, int(df_max.iloc[selection].frequency), df_max.iloc[selection].buy_tresh, df_max.iloc[selection].sell_tresh, 1, selection='open', debug=1)
        plt.title(str(result))
        # select a frequency and plot heat map
        plt.figure()
        sns.heatmap(df[df.frequency == 2].pivot('buy_tresh','sell_tresh','totalasset'))
        # try on a frequency and ratio
        plt.figure()
        result=price_analysis(Data, 2, 0.05, -0.01, 1, selection='open', debug=1)
        # averaging on a frequency
        plt.figure()
        result=df.pivot_table('gainpercent', 'sell_tresh','buy_tresh', np.mean)
        sns.heatmap(result, xticklabels=result.columns.values.round(3), yticklabels=result.index.values.round(3))





