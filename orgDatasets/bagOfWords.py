'''
Created on Nov 23, 2016

@author: chris
'''

import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

raw_data = pd.read_csv('Combined_News_DJIA.csv')

# print(raw_data.head())
# print(raw_data.iloc[277])
# nulls = pd.isnull(raw_data.iloc[277, 24])
# if nulls:
#     print("it is null")
# for i in range(len(nulls)):
#     if nulls[i]:
#         print('NULL!!!')
#     else:
#         print("Not null...")
# print(len(raw_data.index))
# print(len(raw_data))

all_words = set()

# count up the raw words as a list of dictionaries per day
daily_words = []
dates = []
labels = []
for i in range(len(raw_data)):
    daily = {}
    dates.append(raw_data.iloc[i,0])
    labels.append(raw_data.iloc[i,1])
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
            
            # keep track of a set of unique words        
            all_words.update(words)
    daily_words.append(daily)
    
print("all words=", len(all_words))

# get grand totals per word
total_word_count = {}
for d in daily_words:
    for key in d.keys():
        if key in total_word_count:
            total_word_count[key] += d[key]
        else:
            total_word_count[key] = d[key]

# get rid of words that don't appear often to whittle down overall size            
top_words = []
for w in total_word_count.keys():
    # threshold can be adjusted higher if we want - 5 got rid of ~2/3 of the words
    if total_word_count[w] >= 5:
        top_words.append(w)
        
print("top words=", len(top_words))

# now purge low quality words from the dailys
daily_top_words = []
for d in daily_words:
    d_top = {}
    for key in d.keys():
        if key in top_words:
            d_top[key] = d[key]
            
    daily_top_words.append(d_top)
    
# add dates and labels
for i in range(len(daily_top_words)):
    daily_top_words[i]['Date'] = dates[i]
    daily_top_words[i]['Label'] = labels[i]
    
# add features from sentiment analysis
sentimentA = pd.read_csv('../data/stockSentimentA.csv')
positivity = []
negativity = []
changeA = []
for i in range(len(sentimentA)):
    positivity.append(sentimentA.iloc[i,1])
    negativity.append(sentimentA.iloc[i,2])
    changeA.append(sentimentA.iloc[i,3])
    
sentimentB = pd.read_csv('../data/stockSentimentB.csv')
pleasantness = []
attention = []
sensitivity = []
aptitude = []
polarity = []
changeB = []
for i in range(len(sentimentB)):
    pleasantness.append(sentimentB.iloc[i,1])
    attention.append(sentimentB.iloc[i,2])
    sensitivity.append(sentimentB.iloc[i,3])
    aptitude.append(sentimentB.iloc[i,4])
    polarity.append(sentimentB.iloc[i,5])
    changeB.append(sentimentB.iloc[i,6])

for i in range(len(daily_top_words)):
    daily_top_words[i]['Positivity'] = positivity[i]
    daily_top_words[i]['Negativity'] = negativity[i]
    daily_top_words[i]['Stock Change A'] = changeA[i]
    daily_top_words[i]['Pleasantness'] = pleasantness[i]
    daily_top_words[i]['Attention'] = attention[i]
    daily_top_words[i]['Sensitivity'] = sensitivity[i]
    daily_top_words[i]['Aptitude'] = aptitude[i]
    daily_top_words[i]['Polarity'] = polarity[i]
    daily_top_words[i]['Stock Change B'] = changeB[i]

# write it all to CSV    
with open('wordcounts.csv', 'w', newline='') as csvfile:
    header = ['Date', 'Label', 'Stock Change A', 'Stock Change B', 
              'Positivity', 'Negativity', 'Pleasantness', 'Attention', 
              'Sensitivity', 'Aptitude', 'Polarity'] + sorted(top_words)
    writer = csv.DictWriter(csvfile, fieldnames=header, restval='0')
    writer.writeheader()
    writer.writerows(daily_top_words)
        

