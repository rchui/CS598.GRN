""" grn.py

Main body of grn process.
"""

import csv
import math
import copy
import argparse
import itertools
import time
# import nltk
from scipy import stats
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import svm
import numpy as np

FLAGS = None
OUTCOME = 'GLM'
SNPS = 10
ALPHA = 0.05

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self).__init__(*args, **kwargs)

    def fit(self, X, y, sample_weight=None):
        self = super(LinearRegression, self).fit(X, y, sample_weight=None)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
            for i in range(sse.shape[0])
            ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

def print_log(start_time, title, message):
    """ Prints out a log of the current process. """
    print('%s - ' % timing(start_time) + title + message, '\n')

def build_set():
    """ Builds a set of the samples in the expression data. """
    samples_set = None
    with open(FLAGS.expression) as exp_file:
        samples_set = set([element for element in exp_file.readline().split('\t')][1:])
    return samples_set

def get_outcomes(sample_set):
    """ Gets the outcomes from the outcome file. """
    out_data = None
    with open(FLAGS.outcome) as out_file:
        # For every row in the outcome file, check if the sample is in sample_set
        # If the cancer is glioblastoma multiforme set to 1 else set to 0
        out_data = [1 if row[18] == 'glioblastoma multiforme' else 0 for row in csv.reader(out_file, delimiter='\t') if row[0] in sample_set]
    return np.asarray(out_data)

def get_expression():
    """ Get the expression data from the expression file. """
    exp_data = None
    with open(FLAGS.expression) as exp_file:
        tsv_in = csv.reader(exp_file, delimiter='\t')
        tsv_in = itertools.islice(tsv_in, SNPS + 1)

        # For each element in the CSV set all elements 'NA' to 0.
        exp_data = np.asarray([['0' if element == 'NA' else element for element in row] for row in tsv_in])
        exp_samples = exp_data[0][1:]
        # Remove the sample names
        exp_data = exp_data[1:]
        exp_names = np.asarray([row[0] for row in exp_data])
        # Delete the snp names
        exp_data = np.delete(exp_data, 0, 1).astype(float)

        # Match the expression sample order to the sample order in the file.
        exp_data_out = []
        for row in exp_data:
            exp_samples_copy = copy.deepcopy(exp_samples)
            _, row = (list(t) for t in zip(*sorted(zip(exp_samples_copy, row))))
            exp_data_out.append(row[:-1])
        exp_data_out = np.asarray(exp_data_out)

        return exp_data_out, exp_names

def name_combinations(name_list):
    """ Compute name combinations. """
    size = len(name_list)
    result = copy.deepcopy(name_list)
    for i, name_1 in enumerate(name_list):
        for name_2 in name_list[i + 1:]:
            result = np.append(result, name_1 + '::' + name_2)
    for i in range(size):
        result[i] = name_list[i] + '::' + OUTCOME
    return result

def data_combinations(data_list):
    """ Compute data combinations. """
    result = copy.deepcopy(data_list)
    for i, row_1 in enumerate(data_list):
        for row_2 in data_list[i + 1:]:
            result = np.concatenate((result, [np.multiply(row_1, row_2)]))
    return result

def timing(start_time):
    """ Gets the current timing of the program. """
    start = start_time
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def stepwise_regression(exp_data, out_data, start_time):
    """ Performs stepwise regresion. """
    iteration = 1
    lin_reg = LinearRegression()
    snp_set = set()
    y = np.transpose(np.asarray([out_data]))
    last_p = []
    while True:
        index_p = None
        min_p = 1000000

        if iteration == 1: # No backwards elimination
            for i, row in enumerate(exp_data):
                X = np.transpose(np.asarray([row]))
                lin_reg.fit(X, y)
                if lin_reg.p < min_p:
                    min_p = lin_reg.p[0][0]
                    index_p = i
            if min_p > ALPHA:
                break
            else:
                snp_set.add(index_p)
        else: # Foward and backward step
            x_indexes = []
            x_values = []
            chosen_p = []
            size = len(snp_set)

            # Collect indexes and values from snp_set
            for index in snp_set:
                x_indexes.append(index)
                x_values.append(exp_data[index])

            # Perform forward step
            for i, row in enumerate(exp_data):
                if i not in snp_set:
                    x_values.append(row)
                    X = np.transpose(np.asarray(x_values))
                    lin_reg.fit(X, y)
                    if lin_reg.p[0][-1] < min_p:
                        chosen_p = lin_reg.p
                        min_p = lin_reg.p[0][-1]
                        index_p = i
                    x_values = x_values[:-1]

            # Check break condition with bonferonni corrected ALPHA
            if min_p > ALPHA / size:
                break
            else: # Perform FDR backwards elimination
                last_p = copy.deepcopy(chosen_p)
                snp_set.add(index_p)
                size = len(snp_set)
                chosen_p = chosen_p[0][:-1]
                chosen_p, x_indexes = (list(t) for t in zip(*sorted(zip(chosen_p, x_indexes))))
                for i, p_value in enumerate(chosen_p):
                    if p_value > i * ALPHA / size:
                        snp_set.remove(x_indexes[i])
        # print(min_p, snp_set)

        print_log(start_time, 'Iteration %s' % str(iteration), ' - %s' % min_p + ' - %s' % len(snp_set))
        iteration += 1
    print_log(start_time, 'Iteration %s' % str(iteration), ' - %s' % min_p + ' - %s' % len(snp_set))
    return snp_set, last_p[0]

def build_adjacency(snp_set, last_p, exp_names):
    """ Builds the adjacency graph. """
    model_names = {}
    name_position = {}
    name_score = {}

    # Gather the name corresponding position and score
    # Build adjacency graph
    for i, snp in enumerate(snp_set):
        split_names = exp_names[snp].split('::')
        name_position[split_names[0] + '::' + split_names[1]] = snp
        name_position[split_names[1] + '::' + split_names[0]] = snp
        name_score[split_names[0] + '::' + split_names[1]] = last_p[i]
        name_score[split_names[1] + '::' + split_names[0]] = last_p[i]
        for name_1 in split_names:
            for name_2 in split_names:
                if name_1 != name_2:
                    if name_1 in model_names.keys():
                        model_names[name_1].append((name_2, last_p[i]))
                    else:
                        model_names[name_1] = [(name_2, last_p[i])]
    return model_names, name_position, name_score

def dpi_elimination(model_names, name_position, name_score, snp_set):
    """ Uses the adjacency graph to eliminate cycles. """
    first_list = model_names[OUTCOME]
    first_name = OUTCOME
    for second in first_list: # Check all first elements
        second_list = model_names[second[0]]
        second_name = second[0]
        for third in second_list: # Check all second elements
            third_list = model_names[third[0]]
            third_name = third[0]
            for fourth in third_list: # Check all third elements
                fourth_name = fourth[0]
                if first_name == fourth_name: # Check all fourth elements
                    first_transition = first_name + '::' + second_name
                    second_transition = second_name + '::' + third_name
                    third_transition = third_name + '::' + fourth_name

                    # Eliminate cycles by removing lowest of cycles
                    if name_score[first_transition] > name_score[second_transition]:
                        if name_score[second_transition] > name_score[third_transition]:
                            snp_set.discard(name_position[third_transition])
                        else:
                            snp_set.discard(name_position[second_transition])
                    else:
                        if name_score[first_transition] > name_score[third_transition]:
                            snp_set.discard(name_position[third_transition])
                        else:
                            snp_set.discard(name_position[first_transition])
    return snp_set

def main():
    """ Main body of grn.py. """
    start_time = time.time()
    print()

    sample_set = build_set()
    print_log(start_time, 'Built Sample Set', '')

    #clf = sklearn.naive_bayes.GaussianNB()

    #kf = cross_validation.KFold(n_folds = 10)
    #sum = 0

    #for train, test in kf:
    #    print ('TRAIN:', train, 'TEST:', test)

        #train_data = sample_set(train)
        #test_data = sample_set(test)

    kf = KFold(n_splits = 2)

    #for train, test in kf.split(get_expression()):
        #print ('TRAIN:', train, 'TEST:', test)

    out_data = get_outcomes(sample_set)
    exp_data, exp_names = get_expression()
    print_log(start_time, 'Data Ingested', '')

    print ('exp_names:', exp_names, '\n exp_data', exp_data)
    print ('length of exp_names:', str(len(exp_names)))
    print ('length of exp_data:', str(len(exp_data)))

    for train, test in kf.split(exp_data, exp_names):
        print ('TRAIN:', train, 'TEST:', test)
        exp_names = name_combinations(exp_names)
        exp_data = data_combinations(exp_data)

        print ('exp_names:', exp_names, '\n exp_data', exp_data)
        print ('length of exp_names:', str(len(exp_names)))
        print ('length of exp_data:', str(len(exp_data)))

        print_log(start_time, 'Combinations Calculated', '')

        snp_set, last_p = stepwise_regression(exp_data, out_data, start_time)
        print_log(start_time, 'Finished Regression', ' - ' + str(len(snp_set)))
        print ('snp_set', snp_set, '\n')

        model_names, name_position, name_score = build_adjacency(snp_set, last_p, exp_names)
        print_log(start_time, 'Built Adjacency Graph', '')

        snp_set = dpi_elimination(model_names, name_position, name_score, snp_set)
        print_log(start_time, 'Finished DPI Elimination', ' - ' + str(len(snp_set)))

        #clf = nltk.NaiveBayesClassifier.train(train_data)
        #clf.fit(train[train_data], test[train_data].predict_proba(train[test_data])
        #sum = sum + nltk.classifier.accuracy(clf, test_data)

        print(snp_set, '\n')

    #print_log(start_time, 'Average Accuracy', '-' + sum/10)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '-e', '--expression',
        type=str,
        default='',
        help='expression data'
    )
    PARSER.add_argument(
        '-o', '--outcome',
        type=str,
        default='',
        help='outcome data'
    )
    FLAGS, _ = PARSER.parse_known_args()
    main()
