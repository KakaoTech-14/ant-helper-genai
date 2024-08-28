# 카카오 주식 로드

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
import subprocess
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

subprocess.run(['python','road_stock.py'])

subprocess.run(['python','read_news.py'])

subprocess.run(['python','add_label.py'])

kakao_newslabel_match_openchange = pd.read_csv('kakao_newslabel_match_openchange.csv')

label_dif = list()

for i in range(len(kakao_newslabel_match_openchange)):
  label_dif.append([kakao_newslabel_match_openchange['title_label'][i], kakao_newslabel_match_openchange['Change'][i]])


ylist = list()

for i in range(len(kakao_newslabel_match_openchange)):
  ylist.append(kakao_newslabel_match_openchange['Open'][i])

weight_seed = len(kakao_newslabel_match_openchange)

def weighted_sentiment_average(sentiments, weights):
    """
    감정 점수의 가중 평균을 계산하는 함수.

    Parameters:
    sentiments (list of float): 각 기사의 감정 점수 리스트 (예: [1, -1, 0])
    weights (list of float): 각 기사의 가중치 리스트 (예: [0.5, 1.0, 0.8])

    Returns:
    float: 가중 평균 감정 점수
    """
    if len(sentiments) != len(weights):
        raise ValueError("감정 점수 리스트와 가중치 리스트의 길이가 일치해야 합니다.")

    weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("총 가중치의 합이 0이 될 수 없습니다.")

    return weighted_sum / total_weight

# 예제 데이터
# sentiments = [1, -1, 0, 1, -1]  # 예시 감정 점수 리스트
sentiments = kakao_newslabel_match_openchange['title_label']  # 예시 감정 점수 리스트
weights = np.random.rand(weight_seed) # 예시 가중치 리스트
#weights = kakao_news_df['title_weight']  # 예시 가중치 리스트

# 가중 평균 계산
weighted_avg = weighted_sentiment_average(sentiments, weights)

print(f"가중 평균 감정 점수: {weighted_avg}")

# 예제 데이터 (감정 점수와 주가 변화량, 다음날 주가)
# 실제로는 더 많은 데이터를 사용해야 합니다.

X = np.array(
    label_dif
)  # 예: [weighted_avg, stock_change]
y = np.array(ylist)  # 예: next_day_stock_price

# 데이터셋을 학습용과 테스트용으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 초기화 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
predicted_price = model.predict(X_test)

print("테스트 데이터에 대한 실제 주가:", y_test)
print("예측된 주가:", predicted_price)


from sklearn.metrics import mean_squared_error, r2_score

# R² 및 MSE 계산
r2 = r2_score(y_test, predicted_price)
mse = mean_squared_error(y_test, predicted_price)

print(f"R²: {r2}")
print(f"평균 제곱 오차(MSE): {mse}")