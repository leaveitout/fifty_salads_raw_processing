#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys


def add_flow_dataset(*args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate and add the mean and var for each split")
    parser.add_argument("input", help="The input h5 file.", type=str)
    parser.add_argument("dataset", help="Dataset directory.", type=str)
    args = parser.parse_args()

    input_loc = os.path.abspath(args.input)

    if not os.path.isfile(input_loc):
        print("The specified input file {} does not exist.".format(input_loc))
        sys.exit(1)

    try:
        add_mean_and_var(input_loc)
    except ValueError as e:
        print("Error encountered: {}".format(e))
        sys.exit(1)
