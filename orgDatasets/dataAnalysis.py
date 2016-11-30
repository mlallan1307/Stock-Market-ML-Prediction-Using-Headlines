'''
The purpose of this is to:
  give some examples of negative and positive news headlines
'''

import numpy as np
import copy

from makeDatasets import load_sentiment, load_newsReddit_senticnet


def headline_value(data):
  rtn = 0
  rtn += data[0] * 0.00469442  # positive
  rtn += data[1] * -0.00502836 # negative
  return rtn


def headline_value_sentic(data):
  rtn = 0
  rtn += data[0] * -0.02790178 # pleasantness
  rtn += data[1] * 0.00574016  # attention
  rtn += data[2] * 0.02935138  # sensitivity
  rtn += data[3] * -0.01584717 # aptitude
  rtn += data[4] * -0.01867028 # polarity
  return rtn


def sentiment_per_headline(news, sentiment):
  rtn = []
  for date, headlines in news.items():
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
      tmp = [headline_value([pos, neg])]
      tmp.extend([n, date, pos, neg])
      rtn.append(tmp)
  return rtn


def sentiment_per_headline_sentic(news):
  import json
  loaded = {}
  with open('senticnet_headline_values.json', 'r') as fh:
    loaded = json.load(fh)

  rtn = []
  rtnd = {}
  for date, headlines in news.items():
    if date not in loaded:
      print(date, "not in json!")
      return
    for i, hl in enumerate(headlines):
      values = [headline_value_sentic(loaded[date][i])]
      values.extend([hl, date, list(loaded[date][i])])
      rtn.append(values)
      if date not in rtnd:
        rtnd[date] = values[0]
      else:
        rtnd[date] += values[0]
  return rtn, rtnd


def high_low_headlines(data):
  keep = 5

  dCopy = sorted(data, key=(lambda x: x[0]))

  high = list(dCopy[-keep:])
  high.reverse()
  low  = list(dCopy[:keep])

  return (high, low)


def print_headlines(high, low):
  print("Best Headlines")
  for i, x in enumerate(high):
    print(i+1, '->', ' '.join([str(y) for y in x]))

  print()
  print("Worst Headlines")
  for i, x in enumerate(low):
    print(i+1, '->', ' '.join([str(y) for y in x]))


def load_data(dataFile):
  X = []
  y = []
  with open(dataFile) as handle:
    for line in handle:
      if line.startswith('#'):
        continue
      split = line.strip().split(',')[1:]
      y.append(int(split[-1]))
      X.append(np.array(split[:-1], dtype=float))
  return np.array(X), np.array(y)


def load_stock():
  rtn = []
  rtnd = {}
  news_reddit = load_newsReddit_senticnet()
  with open('DJIA_table.csv', 'r') as fh:
    for line in fh:
      if line.startswith("Date"):
        continue
      ls = line.strip().split(',')
      if ls[0] not in news_reddit:
        continue
      rtn.append(float(ls[4]) - float(ls[1]))
      rtnd[ls[0]] = (float(ls[4]) - float(ls[1]))
  return rtn, rtnd


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def label_histogram():
  y, _ = load_stock()
  y1 = [x for x in y if x>0]
  y2 = [x for x in y if x<=0]
  print("High pct:", len(y1)/len(y)) # 0.5414781297134238
  print("Low  pct:", len(y2)/len(y)) # 0.45852187028657615
  # the histogram of the data
  n, bins, patches = plt.hist([y2, y1], 50, normed=1, histtype='stepfilled', color=['red', 'green'], label=['Fall', 'Rise'])

  # add a 'best fit' line
  #y = mlab.normpdf( bins, mu, sigma)
  #l = plt.plot(bins, y, 'r--', linewidth=1)

  plt.xlabel('Daily Stock Market Change (Points)')
  plt.ylabel('Frequency')
  plt.title('Stock Market change Distribution')
  plt.xlim([-1000, 1000])

  plt.legend()
  plt.show()


def senti_scatter():
  _, stockDict = load_stock()
  news_reddit = load_newsReddit_senticnet()
  _, sentiDict = sentiment_per_headline_sentic(news_reddit)
  # get sentiment summary for the day
  # get list of stock change for the day

  x = []
  y = []
  for date, stock in stockDict.items():
    if date not in sentiDict:
      print ("senti_scatter Error")
      return
    senti = sentiDict[date]
    x.append(stock)
    y.append(senti)

  plt.scatter(normalize(x),normalize(y))
  plt.xlim([-0.15, 0.15])
  plt.ylim([-0.15, 0.15])
  plt.show()


if __name__ == "__main__":
  option = 3
  news_reddit = load_newsReddit_senticnet()
  if option == 1:
    senti = load_sentiment('SentiWordNet.csv')
    news_senti = sentiment_per_headline(news_reddit, senti)
    h, l = high_low_headlines(news_senti)
    print_headlines(h, l)
  elif option == 2:
    news_senti, _ = sentiment_per_headline_sentic(news_reddit)
    h, l = high_low_headlines(news_senti)
    print_headlines(h, l)
  elif option == 3:
    # graphs
    #label_histogram()
    senti_scatter()
