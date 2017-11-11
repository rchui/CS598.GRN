""" grn.py

Main body of grn process.
"""

import sys
import csv
import copy
import argparse
import pprint
import multiprocessing
import itertools
from sklearn import linear_model
import numpy as np

FLAGS = None

def build_set():
    samples_set = None
    with open(FLAGS.expression) as exp_file:
        samples_set = set([element for element in exp_file.readline().split('\t')][1:])
    return samples_set

def get_outcomes(sample_set):
    out_data = None
    with open(FLAGS.outcome) as out_file:
        out_data = [1 if row[18] == 'glioblastoma multiforme' else 0 for row in csv.reader(out_file, delimiter='\t') if row[0] in sample_set]
    return np.asarray(out_data)

def get_expression(sample_set):
    exp_data = None
    with open(FLAGS.expression) as exp_file:
        tsv_in = csv.reader(exp_file, delimiter='\t')
        tsv_in = itertools.islice(tsv_in, 100)

        exp_data = np.asarray([['0' if element == 'NA' else element for element in row] for row in tsv_in])
        exp_samples = exp_data[0][1:]
        exp_data = exp_data[1:]
        exp_names = np.asarray([row[0] for row in exp_data])
        exp_data = np.delete(exp_data, 0, 1).astype(float)

        exp_data_out = []
        for row in exp_data:
            exp_samples_copy = copy.deepcopy(exp_samples)
            _, row = (list(t) for t in zip(*sorted(zip(exp_samples_copy, row))))
            exp_data_out.append(row)
        exp_data_out = np.transpose(np.asarray(exp_data_out))
        # print(exp_data_out, '\n')
        # print(exp_samples, '\n')
        # print(exp_names, '\n')

        return exp_data_out, exp_names

def main():
    print()
    sample_set = build_set()
    out_data = get_outcomes(sample_set)
    exp_data, exp_names = get_expression(sample_set)

    print(out_data, '\n')
    print(exp_data, '\n')
    print(exp_names, '\n')

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
