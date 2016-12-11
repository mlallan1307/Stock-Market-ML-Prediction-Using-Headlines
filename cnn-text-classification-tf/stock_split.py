import csv

with open('./data/stocks/Combined_News_DJIA.csv', 'r') as f:
    reader = csv.reader(f)
    stocks = list(reader)

test = []
train = []
for s in stocks:
    if s[0].startswith('2016') or s[0].startswith('2015'):
        test.append(s)
    elif s[0].startswith('Date'):
        test.append(s)
        train.append(s)
    else:
        train.append(s)

print(stocks[0])
print(train[1])
print(test[1])
print(len(train)/len(stocks))

with open('./data/stocks/training.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train)

with open('./data/stocks/testing.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test)

