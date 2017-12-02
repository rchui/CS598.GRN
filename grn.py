""" grn.py

Main body of grn process.
"""

import sys
import csv
import copy
import argparse
import itertools
import time
from sklearn import linear_model
import numpy as np

FLAGS = None
OUTCOME = 'GLM'
SAMPLES = 10

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
        tsv_in = itertools.islice(tsv_in, SAMPLES)

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

def main():
    start_time = time.time()
    print()
    sample_set = build_set()
    out_data = get_outcomes(sample_set)
    exp_data, exp_names = get_expression()

    print('Data Ingested in %ss' % (time.time() - start_time), '\n')
    # print(out_data, '\n')
    # print(exp_data, '\n')
    # print(exp_names, '\n')
    # print(len(out_data), len(exp_data[0]), len(exp_data), len(exp_names), '\n')

    exp_names = name_combinations(exp_names)
    exp_data = data_combinations(exp_data)

    print('Combinations Calculated in %ss' % (time.time() - start_time), '\n')
    print(exp_names, '\n')
    print(exp_data, '\n')

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
