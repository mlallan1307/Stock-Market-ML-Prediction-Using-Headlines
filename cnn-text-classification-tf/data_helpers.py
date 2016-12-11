import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def clean_sentence(sentence):

    s = CountVectorizer().build_tokenizer()(sentence.lower())
    s = [w for w in s if w not in ENGLISH_STOP_WORDS]
    return " ".join(s).strip()

def load_stock_data_concat(data_file):
    """
    Load the stock news headline data file
    Concat all headlines per day
    :param data_file:
    :return:
    """

    raw_data = pd.read_csv(data_file)
    dates = raw_data.as_matrix(columns=['Date'])
    labels = raw_data.as_matrix(columns=['Label'])
    headlines = raw_data.as_matrix(columns=raw_data.columns[2:])

    #remove 'nan' values
    for i in range(headlines.shape[0]):
        for j in range(headlines.shape[1]):
            if pd.isnull(headlines[i,j]):
                headlines[i,j] = ""

    positive_examples = []
    negative_examples = []
    for i in range(len(labels)):
        if labels[i,0] == 1:
            positive_examples.append(" ".join(headlines[i]))
        else:
            negative_examples.append(" ".join(headlines[i]))

    x_text = positive_examples + negative_examples
    x_text = [clean_sentence(s) for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_stock_data_singles(data_file):
    """
    Load each headline individually
    :param data_file:
    :return:
    """
    raw_data = pd.read_csv(data_file)
    dates = raw_data.as_matrix(columns=['Date'])
    labels = raw_data.as_matrix(columns=['Label'])
    headlines = raw_data.as_matrix(columns=raw_data.columns[2:])

    #remove 'nan' values
    for i in range(headlines.shape[0]):
        for j in range(headlines.shape[1]):
            if pd.isnull(headlines[i,j]):
                headlines[i,j] = ""

    positive_examples = []
    negative_examples = []
    for i in range(len(labels)):
        if labels[i,0] == 1:
            # positive_examples.append(" ".join(headlines[i]))
            for h in headlines[i]:
                positive_examples.append(h)
        else:
            # negative_examples.append(" ".join(headlines[i]))
            for h in headlines[i]:
                negative_examples.append(h)

    x_text = positive_examples + negative_examples
    x_text = [clean_sentence(s) for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_stock_with_days(data_file):
    raw_data = pd.read_csv(data_file)
    dates = raw_data.as_matrix(columns=['Date'])
    labels = raw_data.as_matrix(columns=['Label'])
    headlines = raw_data.as_matrix(columns=raw_data.columns[2:])

    x_text = []
    y = []
    hl_dates = {}
    # remove 'nan' values
    for i in range(headlines.shape[0]):
        hl_dates[dates[i,0]] = []
        for j in range(headlines.shape[1]):
            if pd.isnull(headlines[i, j]):
                headlines[i, j] = ""

            x_text.append(headlines[i,j])

            # get the index of the headline just added to x_text
            offset = (i * headlines.shape[1]) + j
            hl_dates[dates[i,0]].append(offset)
        if labels[i, 0] == 1:
            y.append([0, 1])
        else:
            y.append([1, 0])

    x_text = [clean_sentence(s) for s in x_text]
    return [x_text, np.asarray(y), hl_dates]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
