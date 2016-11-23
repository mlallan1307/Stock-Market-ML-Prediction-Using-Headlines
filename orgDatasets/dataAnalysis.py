'''
The purpose of this is to:
  give some examples of negative and positive news headlines
'''

import numpy as np

from makeDatasets import load_sentiment, load_newsReddit_senticnet

def sentiment_per_headline(news, sentiment):
  rtn = {}
  for date, headlines in news.items():
    tmp = []
    for n in headlines:
      words = {}
      w = n.split()
      for cw in w:
        cw = cw.lower()
        if cw in words:
          words[cw] += 1
        else:
          words[cw] = 1

      pos = 0
      neg = 0
      for word, freq in words.items():
        if word in sentiment:
          pos += (sentiment[word][0])*freq
          neg += (sentiment[word][1])*freq
      tmp.append([pos, neg, n])
    rtn[date] = tmp
  return rtn


def high_low_headlines(data):
  keep = 5

  high = []
  low = []
  for date, v in data.items():
    for headline in v:
      pos = headline[0]
      neg = headline[1]
      s = pos - neg
      txt = headline[2]
      if len(high) < keep:
        high.append(list(headline))
        continue
      if len(low) < keep:
        low.append(list(headline))
        continue
      i = -1
      for idx, k in enumerate(high):
        if k[0] - k[1] < s:
          # keep looping until lowest found
          if i == -1 or high[i][0] - high[i][1] > k[0] - k[1]:
            i = idx
      if i != -1:
        high[i] = list(headline)

      i = -1
      for idx, k in enumerate(low):
        if k[0] - k[1] > s:
          # keep looping until lowest found
          if i == -1 or low[i][0] - low[i][1] < k[0] - k[1]:
            i = idx
      if i != -1:
        low[i] = list(headline)
  return (high, low)


def sort_high_low_headlines(high, low):
  h = [[(x[0]-x[1]), list(x)] for x in high]
  l = [[(x[0]-x[1]), list(x)] for x in low]

  h.sort(key=lambda x: x[0])
  h.reverse()

  l.sort(key=lambda x: x[0])

  return [x[1] for x in h], [x[1] for x in l]


def print_headlines(high, low):
  print("Best Headlines")
  for i, x in enumerate(high):
    print(i+1, '->', ' '.join([str(y) for y in x]))

  print()
  print("Worst Headlines")
  for i, x in enumerate(low):
    print(i+1, '->', ' '.join([str(y) for y in x]))


if __name__ == "__main__":
  senti = load_sentiment('SentiWordNet.csv')
  news_reddit, prev_day = load_newsReddit_senticnet('RedditNews.csv')
  news_senti = sentiment_per_headline(news_reddit, senti)
  h, l = high_low_headlines(news_senti)
  sh, sl = sort_high_low_headlines(h, l)
  print_headlines(sh, sl)
