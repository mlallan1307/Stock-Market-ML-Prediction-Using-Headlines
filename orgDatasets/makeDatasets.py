
from senticnet4 import senticnet

import numpy as np
import re
import copy
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

def get_lines(fn):
  lines = []
  with open(fn) as handle:
    for l in handle:
      lines.append(l)
  return lines

#############################################################3

def load_sentiment(fn):
  lines = get_lines(fn)
  senti = {}
  for l in lines:
    if not l.startswith('#'):
      ls = l.split(',')
      words = ls[4].split()
      for i in range(len(words)):
        tmp = words[i]
        tmp = re.sub(r'#\d*', '', tmp)
        tmp = ' '.join(tmp.split('_'))
        if tmp in senti:
          senti[tmp][0] += float(ls[2]) # positive
          senti[tmp][1] += float(ls[3]) # negative
          senti[tmp][2] += 1
        else:
          senti[tmp] = [float(ls[2]), float(ls[3]), 1]

  rtn = {}
  for k, v in senti.items():
    pos = v[0]/v[2]
    neg = v[1]/v[2]
    if pos == 0 and neg == 0:
      continue
    rtn[k] = [pos, neg]
  return rtn


def load_sentiment_senticnet():
  senti = {}

  for k, v in senticnet.items():
    revK = ' '.join(k.split('_'))
    data = [float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[7])]
    if revK in senti:
      print("Dup:", revK, k)
      exit()
    senti[revK] = data
  return senti

#############################################################3

def load_stockData(fn):
  lines = get_lines(fn)
  stocks = {}
  for l in lines:
    if not l.startswith('Date'):
      ls = l.split(',')
      stocks[ls[0]] = float(ls[4]) - float(ls[1])
  '''
  values = np.array([])
  # calculate std
  for k, v in stocks.items():
    values = np.append(values, v)
  d0 = np.std(values)*0.1
  d1 = np.std(values)
  '''
  for k, v in stocks.items():
    c = 0
    if v > 0:
      c = 1
    '''
    if v > d1:
      c = 2
    elif v < (-1*d1):
      c = -2
    elif v > d0:
      c = 1
    elif v < (-1*d0):
      c = -1
    '''
    stocks[k] = c
  return stocks

#############################################################3


def load_newsReddit():
  # This method was revised to run as Chris designed it. It's now correlated > 7x better with labels
  raw_data = pd.read_csv('Combined_News_DJIA.csv')
  # count up the raw words as a list of dictionaries per day
  daily_words = {}
  for i in range(len(raw_data)):
    daily = {}
    date = (raw_data.iloc[i,0])
    for j in range(2,27):
      article = raw_data.iloc[i,j]
      if not pd.isnull(article):
        raw_words = CountVectorizer().build_tokenizer()(article.lower())
        # only use significant words
        words = [s for s in raw_words if s not in ENGLISH_STOP_WORDS]
        for word in words:
          if word in daily:
            daily[word] += 1
          else:
            daily[word] = 1
    daily_words[date] = daily
  return daily_words


def load_newsReddit_senticnet():
  raw_data = pd.read_csv('Combined_News_DJIA.csv')
  # count up the raw words as a list of dictionaries per day
  daily_headlines = {}
  for i in range(len(raw_data)):
    date = (raw_data.iloc[i,0])
    label = int(raw_data.iloc[i,1])
    daily = [label]
    for j in range(2,27):
      article = raw_data.iloc[i,j]
      if not pd.isnull(article):
        raw_headline = ' '.join(CountVectorizer().build_tokenizer()(article.lower()))
        #print (article)
        #print (raw_words)
        #input("Continue:")
        daily.append(raw_headline)
    daily_headlines[date] = daily
  return daily_headlines


#############################################################3

def combine_data_thread(headline, sentiment):
  rtn = np.zeros(2)
  for word, value in sentiment.items():
    rtn = np.add(rtn, np.multiply(headline.count(word), value))
  return rtn


def combine_data(sentiment, news):
  from multiprocessing import Pool
  import json
  json_file = 'sentiword_headline_values.json'
  json_dict = {}
  rtn = []
  c = len(news)
  for date, dayNews in news.items():
    v = np.zeros(2)
    label = dayNews[0]
    raw_headlines = []
    print ("{}/{}".format(c, len(news)), date, end='\t')
    c -= 1
    with Pool(processes=18) as TP:
      jobs = []
      for i in range(1, len(dayNews)):
        jobs.append(TP.apply_async(combine_data_thread, (dayNews[i], sentiment)))
      for i in range(len(jobs)):
        res = jobs[i].get()
        v = np.add(v, res)
        raw_headlines.append(list(res))
    json_dict[date] = raw_headlines
    rtn.append([date, str(v[0]), str(v[1]), str(label)])
    print(rtn[-1])
  with open(json_file, 'w') as fh:
    json.dump(json_dict, fh, sort_keys=True)
  return rtn


def combine_data_senticnet_thread(headline, sentiment):
  rtn = np.zeros(5)
  for word, value in sentiment.items():
    rtn = np.add(rtn, np.multiply(headline.count(word), value))
  return rtn


def combine_data_senticnet(sentiment, news):
  from multiprocessing import Pool
  import json
  json_file = 'senticnet_headline_values.json'
  json_dict = {}
  rtn = []
  c = len(stocks)
  for date, dayNews in news.items():
    if date not in stocks:
      continue
    v = np.zeros(5)
    raw_headlines = []
    print ("{}/{}".format(c, len(stocks)), date, end='\t')
    c -= 1
    with Pool(processes=18) as TP:
      jobs = []
      for headline in dayNews:
        jobs.append(TP.apply_async(combine_data_senticnet_thread, (headline, sentiment)))
      for i in range(len(jobs)):
        res = jobs[i].get()
        v = np.add(v, res)
        raw_headlines.append(list(res))
    json_dict[date] = raw_headlines
    rtn.append([date, str(v[0]), str(v[1]), str(v[2]), str(v[3]), str(v[4]), str(stocks[date])])
    print(rtn[-1])
  with open(json_file, 'w') as fh:
    json.dump(json_dict, fh, sort_keys=True)
  return rtn

#############################################################3

def combine_data_prev(sentiment, stocks, news, prev_day):
  rtn = []
  for date, words in news.items():
    if date not in stocks:
      continue
    pos = 0
    neg = 0
    for word, freq in words.items():
      if word in sentiment:
        pos += (sentiment[word][0])*freq
        neg += (sentiment[word][1])*freq
    pos_p = 0
    neg_p = 0
    for word, freq in news[prev_day[date]].items():
      if word in sentiment:
        pos_p += (sentiment[word][0])*freq
        neg_p += (sentiment[word][1])*freq
    rtn.append([date, str(pos), str(neg), str(pos_p), str(neg_p), str(stocks[date])])
  return rtn


def combine_data_prev_senticnet(sentiment, stocks, news, prev_day):
  rtn = []
  for date, dayNews in news.items():
    if date not in stocks:
      continue
    v = np.zeros(5)
    v_p = np.zeros(5)
    for word, value in sentiment.items():
      for story in dayNews:
        v = np.add(v, np.multiply(story.count(word), value))
      for story in news[prev_day[date]]:
        v_p = np.add(v_p, np.multiply(story.count(word), value))
    rtn.append([date,
                str(v[0]), str(v[1]), str(v[2]), str(v[3]), str(v[4]),
                str(v_p[0]), str(v_p[1]), str(v_p[2]), str(v_p[3]), str(v_p[4]),
                str(stocks[date])])
  return rtn

#############################################################3

def print_dataset(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,news positivity,news negativity,stock change\n')
    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      handle.write(','.join(l))
      handle.write('\n')


def print_dataset_prev(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,news positivity,news negativity,yesterday news positivity, yesterday news negativity,stock change\n')
    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      handle.write(','.join(l))
      handle.write('\n')


def print_dataset_prev_only(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,yesterday news positivity, yesterday news negativity,stock change\n')
    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      ln = [l[0]]
      ln.extend(l[3:])
      handle.write(','.join(ln))
      handle.write('\n')

#############################################################3

def print_dataset_senticnet(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,'\
                 'news pleasantness,news attention,news sensitivity,news aptitude,news polarity,'\
                 'stock change\n')
    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      handle.write(','.join(l))
      handle.write('\n')


def print_dataset_prev_senticnet(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,'\
                 'news pleasantness,news attention,news sensitivity,'\
                 'news aptitude,news polarity,'\
                 'prev news pleasantness,prev news attention,prev news sensitivity,'\
                 'prev news aptitude,prev news polarity,'\
                 'stock change\n')

    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      handle.write(','.join(l))
      handle.write('\n')


def print_dataset_prev_only_senticnet(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,'\
                 'prev news pleasantness,prev news attention,prev news sensitivity,'\
                 'prev news aptitude,prev news polarity,'\
                 'stock change\n')
    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      ln = [l[0]]
      ln.extend(l[6:])
      handle.write(','.join(ln))
      handle.write('\n')

#############################################################3

if __name__ == "__main__":
  option = 1

  if option == 1:
    senti = load_sentiment('SentiWordNet.csv')
    news_reddit = load_newsReddit_senticnet()
    combined = combine_data(senti, news_reddit)
    #combined_p = combine_data_prev(senti, stocks, news_reddit, prev_day)

    print_dataset(combined, 'stockSentimentA.csv')
    #print_dataset_prev(combined_p, 'stockSentimentAWithPrev.csv')
    #print_dataset_prev_only(combined_p, 'stockSentimentAOnlyPrev.csv')

  elif option == 2:

    senti = load_sentiment_senticnet()
    news_reddit = load_newsReddit_senticnet()
    # The below takes 7 hours to run
    combined = combine_data_senticnet(senti, news_reddit)
    #combined_p = combine_data_prev_senticnet(senti, stocks, news_reddit, prev_day)

    print_dataset_senticnet(combined, 'stockSentimentB2.csv')
    #print_dataset_prev_senticnet(combined_p, 'stockSentimentBWithPrev.csv')
    #print_dataset_prev_only_senticnet(combined_p, 'stockSentimentBOnlyPrev.csv')
