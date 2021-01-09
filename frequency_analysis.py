import glob
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from scipy import signal
from scipy import fft
import numbers
def frequency_analysis(sig,ratio,debug=0):
    yf=fft.fft(sig)
    N=len(sig)
    xf=np.linspace(0,1/2,N//2)
    YF=2/N*np.abs(yf[0:N//2])
    YF_stab=YF[xf>0.1]
    YF_stab_range=YF_stab.max()-YF_stab.min()
    treshhold=YF_stab_range*ratio
    xf_treshold=xf[np.nonzero(YF> treshhold)[0][-1]]
    if debug:
        plt.figure()
        plt.plot(xf,YF)
        plt.ylim([-treshhold*3,treshhold*3])
        plt.xlim([0,0.1])
        plt.plot([xf_treshold,xf_treshold],[-treshhold*3,treshhold*3])
    return xf_treshold

def randomlize(Data_raw,n=1,minimumsize=168):
    # select n number of radomly chosen historical data
    data_length=len(Data_raw)
    ctr=0
    Data_selected=[]
    while ctr< n:

        startingpoint = random.choice(np.arange(0,data_length-minimumsize))
        endpoint=random.choice(np.arange(startingpoint+minimumsize,data_length))
        Data_selected.append(Data_raw[startingpoint:endpoint])
        ctr=ctr+1
    return Data_selected

def index_analyzer(sig,samplesize,cutoff_f='moving',fratio=2):
    sell_index_A=[]
    buy_index_A=[]
    filtered_signal=[]
    ctr=0
    while ctr<len(sig)-samplesize:
        Data_cropped=sig[ctr:ctr+samplesize]
        if isinstance(cutoff_f, numbers.Number):
            numerator_coeffs, denominator_coeffs = signal.butter(2, cutoff_f)
            filtered = signal.lfilter(numerator_coeffs, denominator_coeffs, Data_cropped)
            if (filtered[-2] > filtered[-1]) and (filtered[-2] > filtered[-3]):
                sell_index_A.append(ctr + samplesize)
            if (filtered[-2] < filtered[-1]) and (filtered[-2] < filtered[-3]):
                buy_index_A.append(ctr + samplesize)
        if cutoff_f == 'moving':
            f=frequency_analysis(Data_cropped,fratio,0)
            numerator_coeffs, denominator_coeffs = signal.butter(2, f)
            filtered = signal.lfilter(numerator_coeffs, denominator_coeffs, Data_cropped)
            if (filtered[-2] > filtered[-1]) and (filtered[-2] > filtered[-3]):
                sell_index_A.append(ctr + samplesize)
            if (filtered[-2] < filtered[-1]) and (filtered[-2] < filtered[-3]):
                buy_index_A.append(ctr + samplesize)

        ctr=ctr+1
    return sell_index_A,buy_index_A



def filter_comp(Data):
    result_but=[]
    ctr=0
    sig = Data['open'].values.astype('float')
    while ctr<len(sig)-1000:
        Data_cropped=sig[ctr:ctr+1000]
        numerator_coeffs, denominator_coeffs = signal.butter(2, 0.02)
        filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, Data_cropped)
        result_but.append(filtered_signal[-1])
        ctr=ctr+1
    plt.figure()
    plt.plot(sig)
    plt.plot(np.arange(1000,len(Data)),result_but)
    plt.plot(signal.lfilter(numerator_coeffs,denominator_coeffs,sig))






def buy_sell(sig,buy_index,sell_index,buy_selltresh,debug=True,debug_f=0.02):
    Euro=1
    Coin=0
    last_trade_price=1
    actualbuy_index=[]
    actualsell_index=[]
    for ctr, i in enumerate(sig):
        if (ctr in buy_index) and (Euro>0) and abs(i-last_trade_price)/last_trade_price>buy_selltresh:
            Coin = Euro / i * (1 - 0.001)
            Euro=0
            actualbuy_index.append(ctr)
            last_trade_price=i
        if (ctr in sell_index) and (Coin>0) and abs(i-last_trade_price)/last_trade_price>buy_selltresh:
            Euro=Coin*i*(1-0.001)
            Coin=0
            actualsell_index.append(ctr)
            last_trade_price=i
    total_asset= Euro + Coin*last_trade_price
    if debug:
        plt.figure()
        ax1=plt.subplot(2,1,1)
        plt.plot(sig)
        plt.plot(actualbuy_index,sig[actualbuy_index],'g*',actualsell_index,sig[actualsell_index],'ro')
        ax2=plt.subplot(2,1,2,sharex=ax1)
        numerator_coeffs, denominator_coeffs = signal.butter(2, debug_f)
        filtered_signal = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)
        plt.plot(filtered_signal)
        plt.plot(actualbuy_index,filtered_signal[actualbuy_index],'g*',actualsell_index,filtered_signal[actualsell_index],'ro')
    return total_asset,actualbuy_index,actualsell_index

def main():
    filename='./history/BTCUSDT.pkl'
    Testing_windowsize=4*7*24 #1 month
    Data=pd.read_pickle(filename)
    ctr=0
    total_asset=[]
    sig = Data['open'].values.astype('float')
    while ctr<len(Data)-5000:


        Data_cropped = sig[ctr:ctr + 5000]
        f_cutoff=frequency_analysis(Data_cropped,0)
        sell_index_A,buy_index_A=index_analyzer(Data_cropped,1000,0.015)
        asset,actualbuy_index,actualsell_index=buy_sell(Data_cropped,buy_index_A,sell_index_A,0.003,debug=False)
        total_asset.append(asset)
        ctr =ctr+5000
        print(ctr)
    print(total_asset)

    filename = './history/ETHUSDT.pkl'
    Data = pd.read_pickle(filename)
    sig = Data['open'].values.astype('float')
    assets=[]
    for f in np.linspace(0.001,0.1,20):
        sell_index_A, buy_index_A = index_analyzer(sig, 1000, f)
        asset, actualbuy_index, actualsell_index = buy_sell(sig, buy_index_A, sell_index_A, 0.001, debug=False)
        assets.append(asset)

    # evaluate the trashholdfound for random choice for bitcoin
    assets=[]
    Data_random=randomlize(sig,n=100,minimumsize=2000)
    for data in Data_random:
        sell_index_A, buy_index_A = index_analyzer(data, 1000, 0.015)
        asset, actualbuy_index, actualsell_index = buy_sell(data, buy_index_A, sell_index_A, 0.001, debug=False)
        assets.append(asset)

    assets=[]
    for fratio in np.linspace(1.2,3,20):
        sell_index_A, buy_index_A = index_analyzer(sig, 1000, 'moving',fratio )
        asset, actualbuy_index, actualsell_index = buy_sell(sig, buy_index_A, sell_index_A, 0.001, debug=False)
        assets.append(asset)

    assets=[]
    files=glob.glob('./history/*.pkl')
    for file in files:
        Data=pd.read_pickle(file)
        sig = Data['open'].values.astype('float')
        sig=sig[0:int(len(sig)/2)]
        f=frequency_analysis(sig,0)
        sell_index_A, buy_index_A = index_analyzer(sig, 1000, 'moving')
        asset, actualbuy_index, actualsell_index = buy_sell(sig, buy_index_A, sell_index_A, 0.001, debug=False)
        assets.append(asset)
