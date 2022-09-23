import ccxt
import time
from datetime import datetime
import pandas as pd
import numpy as np
import math
import larry
import requests


with open("bnc.key") as f:
    lines = f.readlines()
    api_key = lines[0].strip()
    secret  = lines[1].strip()

binance = ccxt.binance(config={
    'apiKey': api_key, 
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
    }
})

binance.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=1)

symbols = binance.symbols

for symbol in symbols:
    if symbol[-4:] == 'BUSD':
        symbols.remove(symbol)

# symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'AVAX/USDT', 'LUNA/USDT', 'DOGE/USDT', 'DOT/USDT', 
#     '1000SHIB/USDT', 'MATIC/USDT', 'LTC/USDT', 'ATOM/USDT', 'LINK/USDT', 'NEAR/USDT', 'UNI/USDT', 'TRX/USDT', 'ALGO/USDT', 'BCH/USDT', 
#     'MANA/USDT', 'XLM/USDT', 'FTM/USDT', 'HBAR/USDT', 'ICP/USDT', 'ETC/USDT', 'EGLD/USDT', 'FIL/USDT', 'KLAY/USDT', 'AXS/USDT', 
#     'THETA/USDT', 'XTZ/USDT', 'XMR/USDT', 'HNT/USDT', 'EOS/USDT', 'FLOW/USDT', 'GALA/USDT', 'AAVE/USDT', 'ONE/USDT', 'GRT/USDT']

# symbols = symbols[:40]

FEE = 0.002
k = 0.5

df_long = pd.DataFrame()
longs = []
long20 = []
shorts = []
short20 = []

day = 20

for symbol in symbols:

    ohlcv = binance.fetch_ohlcv(symbol, timeframe='1d', limit=731+day)
    df = pd.DataFrame(ohlcv, columns = ['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['time']]
    df.set_index('time', inplace=True)

    for i in [2, 3, 5, 7, 10, 14, 21, 30]:
        df[str(i) + 'MA'] = df['close'].rolling(window=i).mean()

    df['bull'] = (df['open'] > df['2MA'].shift(1)) & (df['open'] > df['3MA'].shift(1)) & (df['open'] > df['5MA'].shift(1)) & (df['open'] > df['7MA'].shift(1)) & \
        (df['open'] > df['10MA'].shift(1)) & (df['open'] > df['14MA'].shift(1)) & (df['open'] > df['21MA'].shift(1)) & (df['open'] > df['30MA'].shift(1))
    df['bear'] = (df['open'] < df['2MA'].shift(1)) & (df['open'] < df['3MA'].shift(1)) & (df['open'] < df['5MA'].shift(1)) & (df['open'] < df['7MA'].shift(1)) & \
        (df['open'] < df['10MA'].shift(1)) & (df['open'] < df['14MA'].shift(1)) & (df['open'] < df['21MA'].shift(1)) & (df['open'] < df['30MA'].shift(1))

    df['range'] = df['high'].shift(1) - df['low'].shift(1)
    df['long_target'] = df['open'] + df['range'] * k
    df['short_target'] = df['open'] - df['range'] * k
    df['long_drr'] = np.where((df['high'] > df['long_target']) & ~ df['bear'], 
            (df['close'] / df['long_target']) - FEE, 1)
    # df['long_crr'] = df['long_drr'].cumprod()
    # df['long_max_asset'] = df['long_crr'].cummax()
    # df['long_dd'] = df['long_crr'] / df['long_max_asset'] - 1
    # df['long_mdd'] = df['long_dd'].cummin()


    df['short_drr'] = np.where((df['low'] < df['short_target']) & ~ df['bull'], 
            (df['short_target'] / df['close']) - FEE, 1)
    # df['short_crr'] = df['short_drr'].cumprod()
    # df['short_max_asset'] = df['short_crr'].cummax()
    # df['short_dd'] = df['short_crr'] / df['short_max_asset'] - 1
    # df['short_mdd'] = df['short_dd'].cummin()

    df[symbol[:-5] + '_long'] = df['long_drr']
    df[symbol[:-5] + '_short'] = df['short_drr']
    # df['crr'] = df['drr'].cumprod()
    # df['max_asset'] = df['crr'].cummax()
    # df['dd'] = df['crr'] / df['max_asset'] - 1
    # df['mdd'] = df['dd'].cummin()

    longs.append(df[symbol[:-5] + '_long'])
    long20.append(df[symbol[:-5] + '_long'].shift(1).rolling(window=day).apply(np.prod, raw=True))
    shorts.append(df[symbol[:-5] + '_short'])
    short20.append(df[symbol[:-5] + '_short'].shift(1).rolling(window=day).apply(np.prod, raw=True))

df_long = pd.concat(longs, axis=1)
df_long20 = pd.concat(long20, axis=1)
df_short = pd.concat(shorts, axis=1)
df_short20 = pd.concat(short20, axis=1)

df_long.to_csv('result/long_drr.csv')
df_long20.to_csv('result/long_' + str(day) + 'd_profit.csv')
df_short.to_csv('result/short_drr.csv')
df_short20.to_csv('result/short_' + str(day) + 'd_profit.csv')

f = open('result/best_coins.csv', 'w')
f2 = open('result/result.txt', 'w')
drr_result = []
time_index = []
interval = 0
for time, drr in df_long.iterrows():
    long_coins = []
    short_coins = []
    for symbol in symbols:
        symbol = symbol[:-5]
        long_symbol = symbol + '_long'
        short_symbol = symbol + '_short'
        if np.isnan(df_long20[long_symbol][time]):
            continue
        if len(long_coins) == 0:
            long_coins.append((symbol, df_long20[long_symbol][time]))
        else:
            if len(long_coins) < 10:
                long_coins.append((symbol, df_long20[long_symbol][time]))
            else:
                long_minVal = 10000
                for coin in long_coins:
                    if coin[1] < long_minVal:
                        long_minVal = coin[1]
                        min_long_coin = coin
                
                if df_long20[long_symbol][time] > long_minVal:
                    long_coins.remove(min_long_coin)
                    long_coins.append((symbol, df_long20[long_symbol][time]))

        if np.isnan(df_short20[short_symbol][time]):
            continue
        if len(short_coins) == 0:
            short_coins.append((symbol, df_short20[short_symbol][time]))
        else:
            if len(short_coins) < 5:
                short_coins.append((symbol, df_short20[short_symbol][time]))
            else:
                short_minVal = 10000
                for coin in short_coins:
                    if coin[1] < short_minVal:
                        short_minVal = coin[1]
                        min_short_coin = coin
                
                if df_short20[short_symbol][time] > short_minVal:
                    short_coins.remove(min_short_coin)
                    short_coins.append((symbol, df_short20[short_symbol][time]))

    f.write(str(time))
    f.write(',')
    for coin in long_coins:
        f.write(str(coin[0]))
        f.write(',')
    f.write('\n')
    f.write(',')
    for coin in long_coins:
        f.write(str(coin[1] - 1))
        f.write(',')
    f.write('\n')

    f.write(',')
    for coin in short_coins:
        f.write(str(coin[0]))
        f.write(',')
    f.write('\n')
    f.write(',')
    for coin in short_coins:
        f.write(str(coin[1] - 1))
        f.write(',')
    f.write('\n')
    
    if interval < day:
        interval += 1
    else:
        drr_sum = 0
        coin_num = 0
        for coin in long_coins:
            symbol = coin[0] + '_long'
            try:
                drr_sum += df_long[symbol][time]
                coin_num += 1
            except:
                pass
        for coin in short_coins:
            symbol = coin[0] + '_short'
            try:
                drr_sum += df_short[symbol][time]
                coin_num += 1
            except:
                pass
        try:
            drr_result.append(drr_sum / coin_num)
            time_index.append(time)
        except ZeroDivisionError:
            print('error')
            drr_result.append(np.nan)
            time_index.append(time)

data = {'time': time_index, 'drr': drr_result}
df_result = pd.DataFrame(data)

df_result.set_index('time', inplace=True)

df_result['asset'] = df_result['drr'].cumprod()
df_result['max_asset'] = df_result['asset'].cummax()
df_result['dd'] = df_result['asset'] / df_result['max_asset'] - 1
df_result['mdd'] = df_result['dd'].cummin()
df_result['drr'] = df_result['drr'] - 1

df_result.to_csv('result/univ_auto.csv')

buffer = 'CAGR: ' + str(round((df_result['asset'][-1] ** 0.5 - 1) * 100, 2)) + '\n' + \
         'MDD: ' + str(round(df_result['mdd'][-1] * 100, 2)) + '\n' + \
         'MAR: ' + str(round((df_result['asset'][-1] ** 0.5 - 1) / df_result['mdd'][-1] * (-1), 2)) + '\n'

print(buffer)
f2.write(buffer)

f.close()
f2.close()
