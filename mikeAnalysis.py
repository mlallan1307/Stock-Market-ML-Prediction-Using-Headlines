
import numpy as np
import copy
import random

from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import mean_squared_error, accuracy_score

from matplotlib import pyplot as plt

random.seed()

def split_data_set(data, labels, train_split=0.5, shuffle=True):
  Xtrain = []
  Xtest  = []
  ytrain = []
  ytest  = []
  d = copy.deepcopy(data)
  l = copy.deepcopy(labels)
  if shuffle:
    d, l = shuffle_lists(copy.deepcopy(data), copy.deepcopy(labels))
  nData = int(len(l)*train_split)
  for i, r in enumerate(d[:nData]):
    tmp = copy.deepcopy(r)
    ytrain.append(l[i])
    tmp = np.insert(tmp, 0, 1) # add bias
    Xtrain.append(np.array(tmp))
  for i, r in enumerate(d[nData:]):
    tmp = copy.deepcopy(r)
    ytest.append(l[nData+i])
    tmp = np.insert(tmp, 0, 1) # add bias
    Xtest.append(np.array(tmp))

  return np.array(Xtrain), \
         np.array(ytrain), \
         np.array(Xtest),  \
         np.array(ytest)


def shuffle_lists(l1, l2):
  l1Shuf = []
  l2Shuf = []
  iShuf = list(range(l1.shape[0]))
  random.shuffle(iShuf)
  for i in iShuf:
    l1Shuf.append(l1[i])
    l2Shuf.append(l2[i])
  return (l1Shuf, l2Shuf)


def load_data(dataFile):
  X = []
  y = []
  with open(dataFile) as handle:
    for line in handle:
      if line.startswith('#'):
        continue
      split = line.strip().split(',')[1:]
      y.append(float(split[-1]))
      X.append([float(split[0]), float(split[1])])
  return np.array(X), np.array(y)


def calc_error(label, predict):
  rtn = []
  for i in range(len(label)):
    rtn.append(abs(label[i] - predict[i]))
  return np.array(rtn)


def error_deviation(error, bins):
  rtn = []
  for b in bins:
    tmp = 0
    for e in error:
      if e < b:
        tmp+=1
    rtn.append(float(tmp)/len(error))
  return np.array(rtn)


def show_roc(testErr, trainErr):
  bins = np.arange(0, 800, 2)
  test = error_deviation(testErr, bins=bins)
  train = error_deviation(trainErr, bins=bins)

  plt.plot(bins, test, color="r", label="Test")
  plt.plot(bins, train, color="b", label="Train")
  plt.xlabel('Prediction error')
  plt.ylabel('Percent samples')
  plt.title('SVM Results')
  plt.legend()
  plt.show()


def SVM(dataFile):
  runs = 5

  X, y = load_data(dataFile)

  G = SVR(epsilon=1.0)
  tmpG = 0
  testErr = []
  trainErr = []
  for run in range(runs):
    X_train, y_train, X_test, y_test = split_data_set(X, y, train_split=0.7)
    fit = G.fit(X_train, y_train)
    y_pred_test = fit.predict(X_test)
    y_pred_train = fit.predict(X_train)

    tmpG += mean_squared_error(y_test, y_pred_test)
    testErr = calc_error(y_test, y_pred_test)
    trainErr = calc_error(y_test, y_pred_train)
  mse_G = tmpG/runs
  print("SVM mean squared error:", mse_G)

  show_roc(testErr, trainErr)


def NN(dataFile):
  runs = 5

  X, y = load_data(dataFile)

  C = MLP()
  tmpG = 0
  testErr = []
  trainErr = []
  for run in range(runs):
    X_train, y_train, X_test, y_test = split_data_set(X, y, train_split=0.7)
    fit = C.fit(X_train, y_train)
    y_pred_test = fit.predict(X_test)
    y_pred_train = fit.predict(X_train)

    tmpG += accuracy_score(y_test, y_pred_test)
    testErr = calc_error(y_test, y_pred_test)
    trainErr = calc_error(y_test, y_pred_train)
  mse_C = tmpG/runs
  print("NN mean squared error:", mse_C)



if __name__ == "__main__":
  dataFile = 'stockSentimentA.csv'
  NN(dataFile)
