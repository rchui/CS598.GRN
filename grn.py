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

def build_dict(sample_set):
    out_dict = None
    with open(FLAGS.outcome) as out_file:
        out_dict = {row[0]: 1 if row[18] == 'glioblastoma multiforme' else 0 for row in csv.reader(out_file, delimiter='\t') if row[0] in sample_set}
    return out_dict

def main():
    sample_set = build_set()
    out_dict = build_dict(sample_set)



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
