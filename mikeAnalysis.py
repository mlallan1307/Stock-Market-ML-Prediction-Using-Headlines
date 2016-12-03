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
from multiprocessing import Pool

random.seed()

GEN_KNOWN = {}


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


def SVM_thread(X_train, y_train, X_test, y_test):
  C = SVC(kernel='rbf', C=100)

  #nested_skf = StratifiedKFold(n_splits=folds)
  #y_predict  = cross_val_predict(C, X_train,  y_train, cv=nested_skf, n_jobs=-1)
  y_predict = C.fit(X_train,  y_train).predict(X_test)

  return y_test, y_predict


def SVM(dataset):
  runs = 1
  folds = 12

  #X, y = get_dataset(1) # Bag of words
  #X, y = get_dataset(2) # SentiWord with Bag of words
  #X, y = get_dataset(3) # Sentic with Bag of words
  #X, y = get_dataset(4) # SentiWord
  #X, y = get_dataset(5) # Sentic
  X, y = get_dataset(dataset)

  param_grid = [
    #{'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
    {'C': [1, 10]},
  ]

  # Normalize
  # axis=0 so each feature is normalized for all samples
  XN = normalize(X, axis=0)

  tmpC = 0
  for run in range(runs):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=run)
    folds_y = []
    folds_yPred = []
    with Pool(processes=18) as TP:
      jobs=[]
      for (train_idx, test_idx) in skf.split(X, y):
        X_train, y_train = XN[train_idx], y[train_idx]
        X_test,  y_test  = XN[test_idx],  y[test_idx]
        jobs.append(TP.apply_async(SVM_thread, (X_train, y_train, X_test, y_test)))

      for i in range(len(jobs)):
        res = jobs[i].get()
        folds_y = np.append(folds_y, res[0])
        folds_yPred = np.append(folds_yPred, res[1])

    # Compute accuracy for this fold
    tmpC += accuracy_score(folds_y, folds_yPred)
    print(confusion_matrix(folds_y, folds_yPred))

    print(run+1, "Accuracy:", tmpC/(run+1))

  err_C = tmpC/runs
  print("SVM Accuracy:", err_C)

  #show_roc(testErr, trainErr)


def Gen_convert2Key(p):
  tmp = copy.deepcopy(p)
  tmp[0] = '-'.join([str(x) for x in list(tmp[0])])
  return ('_'.join([str(x) for x in tmp]))


def Gen_get_known(p):
  k = Gen_convert2Key(p)
  rtn = -1
  if k in GEN_KNOWN:
    rtn = GEN_KNOWN[k]
  return rtn


def Gen_set_known(p, fit):
  k = Gen_convert2Key(p)
  global GEN_KNOWN
  if k not in GEN_KNOWN:
    GEN_KNOWN[k] = fit


def Gen_NN_Acc(i, flags, X, y):
  folds = 5

  C = MLP(hidden_layer_sizes=flags[0],
          activation=flags[1],
          alpha=flags[2],
          learning_rate=flags[3],
          max_iter=flags[4])
  skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=i)
  folds_y = []
  folds_yPred = []
  for (train_idx, test_idx) in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    y_predict = C.fit(X_train,  y_train).predict(X_test)

    folds_y = np.append(folds_y, y_test)
    folds_yPred  = np.append(folds_yPred, y_predict)

  # Compute accuracy for this fold
  rtn = accuracy_score(folds_y, folds_yPred)

  return [i, rtn]


def Gen_Fit(pop, X, y):
  fitness = np.zeros(len(pop))
  with Pool(processes=12) as TP:
    jobs=[]
    for i, p in enumerate(pop):
      Known = Gen_get_known(p)
      if (Known != -1):
        fitness[i] = Known
      else:
        jobs.append(TP.apply_async(Gen_NN_Acc, (i, p, X, y)))
    for i in range(len(jobs)):
      res = jobs[i].get()
      fitness[res[0]] = res[1]
      Gen_set_known(pop[res[0]], res[1])
      print(pop[i], '\t', res)
  return fitness


def Gen_Selection(fitness, pop):
  loFit = min(fitness)
  hiFit = max(fitness)

  parents = []
  while len(parents) < 2:
    thresh = random.uniform(loFit, hiFit)
    idx = random.randint(0, len(fitness)-1)
    if fitness[idx] >= thresh:
      parents.append(pop[idx])
  return parents


def Gen_NewPop(p, popSize):
  rtn = [list(p[0]), list(p[1])]
  while len(rtn)<popSize:
    kids = Gen_Crossover(p[0], p[1])
    for i, k in enumerate(kids):
      kids[i] = Gen_Mutate(k)
    rtn.append(kids[0])
    rtn.append(kids[1])

  return rtn[:popSize]


def Gen_Crossover(p1, p2):
  lastFeature = len(p1)-1
  pt = random.randint(1, lastFeature)
  k1 = p1[:pt]
  k1.extend(p2[pt:])
  k2 = p2[:pt]
  k2.extend(p1[pt:])
  return [k1, k2]


def Gen_Mutate(child):
  chance = 0.3
  newKid = []
  for i, feature in enumerate(child):
    if random.random()>chance:
      newKid.append(feature)
    else:
      newFeature = 0
      # hidden_layer_sizes
      if i == 0:
        hl = list(feature)
        nhl = len(hl)
        if nhl > 1:
          # Chance to remove layer
          if random.random()<=(chance/2):
            hl = hl[:-1]
        if nhl < 5:
          # Chance to add layer
          if random.random()<=(chance/2):
            hl.append(hl[-1])
        f = []
        for layer in range(len(hl)):
          newL = hl[layer]
          c = random.choice([10, 20, 50, 100])
          if layer > c:
            if random.random()<=(chance):
              newL -= c
          if layer+c < 500:
            if random.random()<=(chance):
              newL += c
          f.append(newL)
        newFeature = tuple(f)

      # activation
      elif i == 1:
        newFeature = random.choice(['identity', 'logistic', 'tanh', 'relu'])

      # alpha
      elif i == 2:
        f = float(feature)
        if random.random()>0.5:
          f *= 10
        else:
          f /= 10
        newFeature = float(f)

      # learning_rate
      elif i == 3:
        newFeature = random.choice(['constant', 'invscaling', 'adaptive'])

      # max_iter
      elif i == 4:
        f = int(feature)
        if random.random()>0.5:
          f += 100
        elif f>200:
          f -= 100
        newFeature = int(f)

      newKid.append(newFeature)
  return newKid


def NN_genetic_search(dataset):
  X, y = get_dataset(dataset)
  XN = normalize(X, axis=0)

  generations = 100000000
  popSize = 20
  #	hidden_layer_sizes, activation, alpha, learning_rate, max_iter
  starting_features = [(390, 390), 'tanh', 1e-05, 'invscaling', 200]
  pop = []
  bestF = 0
  bestP = []
  for i in range(popSize):
    pop.append(list(starting_features))
  for i in range(generations):
    print("Generation:", i, "Known:", len(GEN_KNOWN), "- Best:", bestP, bestF)
    fitness = Gen_Fit(pop, XN, y)
    for j, f in enumerate(fitness):
      if f>bestF:
        bestF = f
        bestP = list(pop[j])
        print("NEW BEST:", bestF, bestP)
    parents = Gen_Selection(fitness, pop)
    pop = Gen_NewPop(parents, popSize)

def NN_thread(X_train, y_train, X_test,  y_test):
  C = MLP(hidden_layer_sizes=(1000, 100),
          activation='relu',
          learning_rate='adaptive',
          solver='sgd',
          max_iter=300)

  #nested_skf = StratifiedKFold(n_splits=folds)
  #y_predict  = cross_val_predict(C, X_train,  y_train, cv=nested_skf, n_jobs=-1)
  y_predict = C.fit(X_train, y_train).predict(X_test)

  return y_test, y_predict

def NN(dataset):
  runs = 1
  folds = 12

  X, y = get_dataset(dataset)

  # Normalize
  # axis=0 so each feature is normalized for all samples
  XN = normalize(X, axis=0)

  tmpC = 0
  for run in range(runs):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=run)
    folds_y = []
    folds_yPred = []
    with Pool(processes=18) as TP:
      jobs=[]
      for (train_idx, test_idx) in skf.split(X, y):
        X_train, y_train = XN[train_idx], y[train_idx]
        X_test,  y_test  = XN[test_idx],  y[test_idx]
        jobs.append(TP.apply_async(NN_thread, (X_train, y_train, X_test, y_test)))

      for i in range(len(jobs)):
        res = jobs[i].get()
        folds_y = np.append(folds_y, res[0])
        folds_yPred = np.append(folds_yPred, res[1])

    # Compute accuracy for this fold
    tmpC += accuracy_score(folds_y, folds_yPred)
    print(confusion_matrix(folds_y, folds_yPred))

    print(run+1, "Accuracy:", tmpC/(run+1))

  err_C = tmpC/runs
  print("NN Accuracy:", err_C)


def feature_correlation(dataset):
  if dataset == 1:
    X, y, labels = get_dataset(1, labels=True)
  else:
    X, y = get_dataset(dataset)

  pears = []
  idx = []
  for i in range(len(X[0])):
    idx.append(i)
    e = []
    for j in range(len(X)):
      e.append(X[j][i])
    pears.append(np.corrcoef(e, y)[0][1])
  print(np.array(pears))
  if dataset == 1:
    orig = list(zip(idx, pears))
    orig.sort(key=lambda x: abs(x[1]))
    print("Least Correlation")
    for i in range(5):
      f = orig[i]
      print('  ', labels[f[0]], '\t', f[1])

    print("\nMost Correlation")
    for i in range(5):
      f = orig[-1*(i+1)]
      print('  ', labels[f[0]], '\t', f[1])
  print("Sum:", np.sum(np.absolute(np.array(pears))))
  print("Average:", np.average(np.absolute(np.array(pears))))


def get_dataset(i, labels=False):
  import csv
  y = []
  bag = []
  sentiWord = []
  sentic = []
  label = []
  dataSetSplit = 6
  firstWord = 11
  with open('orgDatasets/wordcounts.csv', 'r') as fh:
    lines = csv.reader(fh)
    for n, l in enumerate(lines):
      if n != 0:
        y.append(int(l[1]))
        bag.append(list(l[firstWord:]))
        sentiWord.append(list(l[4:dataSetSplit]))
        sentic.append(list(l[dataSetSplit:firstWord]))
      else:
        label = list(l[firstWord:])

  for n in range(len(y)):
    sentiWord[n].extend(bag[n])
    sentic[n].extend(bag[n])

  if i == 1:
    # Bag of words only
    if labels:
      return np.array(bag, dtype=float), np.array(y), np.array(label)
    else:
      return np.array(bag, dtype=float), np.array(y)

  elif i == 2:
    # SentiWordNet + bag of words
    return np.array(sentiWord, dtype=float), np.array(y)

  elif i == 3:
    # SenticNet + bag of words
    return np.array(sentic, dtype=float), np.array(y)

  elif i == 4:
    # SentiWordNet
    return load_data('data/stockSentimentA.csv')

  elif i == 5:
    # SenticNet
    return load_data('data/stockSentimentB.csv')

  else:
    print("Inavlid entry: get_dataset")
    return



if __name__ == "__main__":
  option = 4

  if option == 1:
    SVM(4)

  elif option == 2:
    NN(4)

  elif option == 3:
    NN_genetic_search(3)

  elif option == 4:
    print("Bag of words\n")
    feature_correlation(1)
    print('-'*50)
    print("SentiWordNet\n")
    feature_correlation(4)
    print('-'*50)
    print("SenticNet\n")
    feature_correlation(5)
