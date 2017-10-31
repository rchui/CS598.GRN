""" grn.py

Main body of grn process.
"""

import sys
import csv
import argparse
import multiprocessing
from sklearn import linear_model
import numpy as np

FLAGS = None

def main():
    """ Main body of the grn process. """
    with open(FLAGS.expression) as expression_file:
        tsv_in = csv.reader(expression_file, delimiter='\t')
        expression_data = np.asarray([np.asarray(row) for row in tsv_in if 'NA' not in row])
        expression_data = np.delete(expression_data, 0, 0)
        expression_names = np.asarray([element[0] for element in expression_data])
        expression_data = np.delete(expression_data, 0, 1).astype(float)

        print(expression_names)
        print(expression_data)

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '-e', '--expression',
        type=str,
        default='',
        help='expression data'
    )
    PARSER.add_argument(
        '-p', '--phenotype',
        type=str,
        default='',
        help='phenotype data'
    )
    FLAGS, _ = PARSER.parse_known_args()
    main()
