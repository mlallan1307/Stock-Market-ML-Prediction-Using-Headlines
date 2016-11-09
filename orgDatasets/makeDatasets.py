
import numpy as np
import re
import copy

def get_lines(fn):
  lines = []
  with open(fn) as handle:
    for l in handle:
      lines.append(l)
  return lines


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


def load_stockData(fn):
  lines = get_lines(fn)
  stocks = {}
  for l in lines:
    if not l.startswith('Date'):
      ls = l.split(',')
      stocks[ls[0]] = float(ls[4]) - float(ls[1])
  return stocks


def load_newsReddit(fn):
  lines = get_lines(fn)
  news = {}
  lines_new = []
  lc = ''
  for l in lines:
    if not l.startswith('Date'):
      if re.search(r'^20\d{2}-\d{2}-\d{2}', l):
        lines_new.append(re.sub('\n', '', l))
      else:
        lines_new[-1] += re.sub('\n', '', l)

  for l in lines_new:
    ls = l.split(',')
    n = l[len(ls[0])+1:]
    rm_list = r'"|\'|\.|,|:|\t|\||;|\?|\!|\$|^b|\[|\]|\(|\)'
    sp_list = r'\s+-\s+|-\s+|--+'
    n = n.strip()
    n = re.sub(rm_list, '', n)
    n = re.sub(sp_list, ' ', n)
    if ls[0] in news:
      news[ls[0]].append(n)
    else:
      news[ls[0]] = [n]

  counts = {}
  for k, v in news.items():
    words = {}
    for n in v:
      w = n.split()
      for cw in w:
        cw = cw.lower()
        if cw in words:
          words[cw] += 1
        else:
          words[cw] = 1
    counts[k] = words
  return counts


def combine_data(sentiment, stocks, news):
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
    rtn.append([date, str(pos), str(neg), str(stocks[date])])
  return rtn


def print_dataset(data, fn):
  with open(fn, 'w') as handle:
    handle.write('#Date,news positivity,news negativity,stock change\n')
    d = copy.deepcopy(data)
    d.sort(key=lambda x: x[0])
    for l in d:
      handle.write(','.join(l))
      handle.write('\n')


if __name__ == "__main__":
  senti = load_sentiment('SentiWordNet.csv')
  stocks = load_stockData('DJIA_table.csv')
  news_reddit = load_newsReddit('RedditNews.csv')
  combined = combine_data(senti, stocks, news_reddit)

  print_dataset(combined, 'stockSentiment.csv')
