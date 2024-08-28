# pip install konlpy
import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

nsmc_train_df = pd.read_csv('./ratings_train.txt', encoding='utf8', sep='\t', engine='python')
nsmc_train_df.head()

nsmc_train_df = nsmc_train_df[nsmc_train_df['document'].notnull()]

# 부정 0 긍정 1
nsmc_train_df['label'].value_counts()
nsmc_train_df['document'] = nsmc_train_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', " ", x))
# nsmc_train_df.head()

nsmc_test_df = pd.read_csv('./ratings_test.txt', encoding='utf8', sep='\t', engine='python')
# nsmc_test_df.head()
nsmc_test_df = nsmc_test_df[nsmc_test_df['document'].notnull()]
# print(nsmc_test_df['label'].value_counts())
nsmc_test_df['document'] = nsmc_test_df['document'].apply(lambda x : re.sub(r'[^ ㄱ-ㅣ가-힣]+', "", x))


okt = Okt()

def okt_tokenizer(text):
    tokens = okt.morphs(text)
    return tokens


tfidf = TfidfVectorizer(tokenizer = okt_tokenizer, ngram_range=(1,2), min_df=3, max_df=0.9)
tfidf.fit(nsmc_train_df['document'])
nsmc_train_tfidf = tfidf.transform(nsmc_train_df['document'])
# print('*** TF-IDF 기반 피처 벡터 생성 ***')

SA_lr = LogisticRegression(random_state = 0)
SA_lr.fit(nsmc_train_tfidf, nsmc_train_df['label'])

params = {'C': [1, 3, 3.5, 4, 4.5, 5]}
SA_lr_grid_cv = GridSearchCV(SA_lr, param_grid=params, cv=3, scoring='accuracy', verbose=1)
SA_lr_grid_cv.fit(nsmc_train_tfidf, nsmc_train_df['label'])
# print(SA_lr_grid_cv.best_params_, round(SA_lr_grid_cv.best_score_, 4))

# 최적 파라미터의 best 모델 저장
SA_lr_best = SA_lr_grid_cv.best_estimator_

# 5~10분정도 소요
# 평가용 데이터의 피처 벡터화 : 실행시간 6분 정도 걸립니다 ☺
nsmc_test_tfidf = tfidf.transform(nsmc_test_df['document'])
# print('*** 평가용 데이터의 피처 벡터화 ***')

test_predict = SA_lr_best.predict(nsmc_test_tfidf)
print('감성 분석 정확도 : ', round(accuracy_score(nsmc_test_df['label'], test_predict), 3))


kakao_news_title_date = pd.read_csv('kakao_news_title_date.csv')
kakao_stock_open_change = pd.read_csv('kakao_stock_open_change.csv')


# 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
data_title_tfidf = tfidf.transform(kakao_news_title_date['title'])
# 2) 최적 파라미터 학습모델에 적용하여 감성 분석
data_title_predict = SA_lr_best.predict(data_title_tfidf)
# 3) 감성 분석 결과값을 데이터 프레임에 저장
kakao_news_title_date['title_label'] = data_title_predict

kakao_news_title_date.rename(columns={'date':'Date'},inplace=True)
kakao_news_title_date['Date'] = pd.to_datetime(kakao_news_title_date['Date'])
kakao_stock_open_change['Date'] = pd.to_datetime(kakao_stock_open_change['Date'])

kakao_newslabel_match_openchange = pd.merge(kakao_news_title_date, kakao_stock_open_change, on='Date')


kakao_newslabel_match_openchange.to_csv('kakao_newslabel_match_openchange.csv',encoding='utf-8-sig',index=True)