# 뉴스 데이터 불러오기 (서울경제)
import requests
from bs4 import BeautifulSoup
title_list = []
date_list = []

#최신~ 61페이지까지 로드
for i in range(1, 70):

  if i == 1:
    url = 'https://www.sedaily.com/NewsList/GD05'
  else:
    url = f'https://www.sedaily.com/NewsList/GD05/New/{i}'


  # 웹 페이지 요청
  response = requests.get(url)
  response.raise_for_status()  # 요청이 성공했는지 확인

  # BeautifulSoup 객체 생성
  soup = BeautifulSoup(response.content, 'html.parser')

  titles = soup.select('.article_tit')
  rel_times = soup.select('.rel_time')
  dates = soup.select('.date')



  for title in titles:
    title_list.append(title.get_text())
  for rel_time in rel_times:
    date_list.append(rel_time.get_text())

  for date in dates:
    date_list.append(date.get_text())


import pandas as pd

df = pd.DataFrame({'title':title_list,'date':date_list})

# '카카오' 들어가는 뉴스 제목만 가져오기

kakao_news_date = []
kakao_news_title = []

for i in range(len(title_list)):
  if '카카오' in title_list[i]:
    kakao_news_title.append(title_list[i])
    kakao_news_date.append(date_list[i])

kakao_news_title_date = pd.DataFrame({'title':kakao_news_title,'date':kakao_news_date})
kakao_news_title_date.to_csv('kakao_news_title_date.csv',encoding='utf-8-sig',index=True)