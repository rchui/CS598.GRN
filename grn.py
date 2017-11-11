""" grn.py

Main body of grn process.
"""

import sys
import csv
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
    return out_data

def get_expression(sample_set):
    exp_list = None
    with open(FLAGS.expression) as exp_file:
        tsv_in = csv.reader(exp_file, delimiter='\t')
        tsv_in = itertools.islice(tsv_in, 100)
        exp_data = np.asarray([row for row in tsv_in])
        exp_samples = exp_data[0][1:]
        exp_data = exp_data[1:]
        exp_names = np.asarray([row[0] for row in exp_data])
        exp_data = np.delete(exp_data, 0, 1)
        print(exp_data, '\n')
        print(exp_samples, '\n')
        print(exp_names, '\n')

def main():
    print()
    sample_set = build_set()
    out_data = get_outcomes(sample_set)
    get_expression(sample_set)

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
