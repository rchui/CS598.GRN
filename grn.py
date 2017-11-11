""" grn.py

Main body of grn process.
"""

import sys
import csv
import argparse
import multiprocessing
import itertools
from sklearn import linear_model
import numpy as np

FLAGS = None

def get_expression():
    """ Reads in the expression data from file.

    Args:
        None

    Returns:
        expression_data: gene  expression data
        expression_names: gene expression names
    """
    with open(FLAGS.expression) as expression_file:
        tsv_in = csv.reader(expression_file, delimiter='\t')
        tsv_in = list(itertools.islice(tsv_in, 100))
        expression_data = np.asarray([np.asarray([element.replace('NA', '0') for element in row]) for row in tsv_in])
        expression_data = np.delete(expression_data, 0, 0)
        expression_names = np.asarray([element[0] for element in expression_data])
        expression_data = np.delete(expression_data, 0, 1).astype(float)
    return expression_data, expression_names

def get_outcome():
    """ Reads in the outcome data from file.

    Args:
        None

    Returns:
        outcome_data: phenotype outcome data
    """
    with open(FLAGS.outcome) as outcome_file:
        tsv_in = csv.reader(outcome_file, delimiter='\t')
        outcome_data = np.asarray([row[1] for row in tsv_in][1:])
        return outcome_data

def main():
    """ Main body of the grn process. """
    print()
    expression_data, expression_names = get_expression()
    outcome_data = get_outcome()

    print(expression_names, '\n')
    print(expression_data, '\n')
    print(outcome_data, '\n')

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
