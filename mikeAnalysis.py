'''
 You need scikit-learn V0.18.x to run this
'''

import numpy as np
import copy
import random

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from matplotlib import pyplot as plt

random.seed()

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
  runs = 2
  folds = 20

  X, y = load_data(dataFile)

  param_grid = [
    #{'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
    {'C': [1, 10]},
  ]

  # Normalize
  # axis=0 so each feature is normalized for all samples
  XN = normalize(X, axis=0)

  tmpC = 0
  for run in range(runs):
    C = SVC(kernel='rbf')
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=run)
    folds_y = []
    folds_yPred = []
    for (train_idx, test_idx) in skf.split(X, y):
      X_train, y_train = XN[train_idx], y[train_idx]
      X_test,  y_test  = XN[test_idx],  y[test_idx]
      nested_skf = StratifiedKFold(n_splits=folds)

      y_predict  = cross_val_predict(C, X_train,  y_train, cv=nested_skf, n_jobs=-1)
      folds_y = np.append(folds_y, y_train)
      folds_yPred  = np.append(folds_yPred, y_predict)

    # Compute accuracy for this fold
    tmpC += accuracy_score(folds_y, folds_yPred)
    print(confusion_matrix(folds_y, folds_yPred))

    print(run+1, "Error:", tmpC/(run+1))

  err_C = tmpC/runs
  print("SVM error:", err_C)

  #show_roc(testErr, trainErr)


def NN(dataFile):
  runs = 2
  folds = 20

  X, y = load_data(dataFile)


  # Normalize
  # axis=0 so each feature is normalized for all samples
  XN = normalize(X, axis=0)

  tmpC = 0
  for run in range(runs):
    C = MLP()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=run)
    folds_y = []
    folds_yPred = []
    for (train_idx, test_idx) in skf.split(X, y):
      X_train, y_train = XN[train_idx], y[train_idx]
      X_test,  y_test  = XN[test_idx],  y[test_idx]
      nested_skf = StratifiedKFold(n_splits=folds)

      y_predict  = cross_val_predict(C, X_train,  y_train, cv=nested_skf, n_jobs=-1)
      folds_y = np.append(folds_y, y_train)
      folds_yPred  = np.append(folds_yPred, y_predict)

    # Compute accuracy for this fold
    tmpC += accuracy_score(folds_y, folds_yPred)
    print(confusion_matrix(folds_y, folds_yPred))

    print(run+1, "Error:", tmpC/(run+1))

  err_C = tmpC/runs
  print("NN error:", err_C)


if __name__ == "__main__":
  dataFile = 'data/stockSentimentBWithPrev.csv'
  SVM(dataFile)
  #NN(dataFile)
