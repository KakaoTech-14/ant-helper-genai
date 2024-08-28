import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import re
import warnings
warnings.filterwarnings(action='ignore')
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

okt = Okt()

def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens


nsmc_train_df = pd.read_csv('./ratings_train.txt', encoding='utf8', sep='\t', engine='python')
nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]

nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))


# TF-IDF 벡터라이저 생성
tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])
#nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])


SA_lr_best = joblib.load("./SA_lr_best.pkl")
## predict = clf.predict(입력값)
tomorrow_stock = joblib.load('./tomorrow_stock.pkl')





today_title_list = []
today_date_list = []

url = 'https://www.sedaily.com/NewsList/GD05'

# 웹 페이지 요청
response = requests.get(url)
response.raise_for_status()  # 요청이 성공했는지 확인

# BeautifulSoup 객체 생성
soup = BeautifulSoup(response.content, 'html.parser')

titles = soup.select('.article_tit')
rel_times = soup.select('.rel_time')
dates = soup.select('.date')



for title in titles:
  today_title_list.append(title.get_text())
for rel_time in rel_times:
  today_date_list.append(rel_time.get_text())

for date in dates:
  today_date_list.append(date.get_text())

  

today_df = pd.DataFrame({'title':today_title_list,'date':today_date_list})


today_kakao_news_date = []
today_kakao_news_title = []

for i in range(len(today_title_list)):
  if '카카오' in today_title_list[i]:
    today_kakao_news_title.append(today_title_list[i])
    today_kakao_news_date.append(today_date_list[i])


today_kakao_news_title_date = pd.DataFrame({'title':today_kakao_news_title,'Date':today_kakao_news_date})

print(today_kakao_news_title_date)


# fitting을 위해 트레이닝 데이터로 fit
# tfidf.fit(today_kakao_news_title_date['title'])
# 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
today_data_title_tfidf = tfidf.transform(today_kakao_news_title_date['title'])

# 2) 최적 파라미터 학습모델에 적용하여 감성 분석
today_data_title_predict = SA_lr_best.predict(today_data_title_tfidf)

# 3) 감성 분석 결과값을 데이터 프레임에 저장
today_kakao_news_title_date['title_label'] = today_data_title_predict




weight_seed = len(today_kakao_news_title_date)

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

# 8/26 뉴스에 대한 데이터
sentiments = today_kakao_news_title_date['title_label']  # 예시 감정 점수 리스트
weights = np.random.rand(weight_seed)
# 가중 평균 계산
weighted_avg = weighted_sentiment_average(sentiments, weights)

print(f"가중 평균 감정 점수: {weighted_avg}")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 테스트 데이터에 대한 예측
input_data = np.array([weighted_avg, weights.item()]).reshape(1, -1)
predicted_price = tomorrow_stock.predict(input_data)
print("예측된 주가:", predicted_price)