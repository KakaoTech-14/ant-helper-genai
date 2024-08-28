import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf


start = (2000, 1, 1)  # 2020년 01년 01월
start = datetime.datetime(*start)
end = datetime.date.today()  # 현재

# yahoo 에서 카카오 불러오기
df = yf.download('035720.KS', start=start, end=end)
df.head(5)
df.tail(5)
df.Close.plot(grid=True)
print("<카카오 ('035720.KS') 주가 Historical Data>\n")
print(df)
print('\n')
print(df.info())

# 거래량이 0인 일자 제거 & 수정시가 데이터만 사용
data = df['Open'][df['Volume'] != 0]
data = data.to_frame()

kakao_dif = []

for i in range(len(data)):
  if i == 0:
    kakao_dif.append(0)
  else:
    kakao_dif.append(data['Open'][i]-data['Open'][i-1])

data['Change'] = kakao_dif

# data.Change.plot(grid=True)

# 해당 시가, 변화량 csv 파일로 저장

data.to_csv('kakao_stock_open_change.csv',encoding='utf-8-sig',index=True)