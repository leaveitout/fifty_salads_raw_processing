#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys

import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder


def dense_to_one_hot(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = np.expand_dims(y, -1)

    encoder = OneHotEncoder(dtype=np.float32, sparse=False)
    y_encoded = encoder.fit_transform(y)
    return y_encoded


def change_dense_to_one_hot(input_h5_path: os.path, labels_key: str):
    with h5py.File(input_h5_path, 'r+') as h5_file:
        labels = h5_file[labels_key]
        labels_one_hot = dense_to_one_hot(labels)

        del h5_file[labels_key]

        h5_file.create_dataset(name=labels_key, data=labels_one_hot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=""
                    "based on the existing fine granularity."
    )

    parser.add_argument("input_h5",
                        help="The input h5 file for the dataset.",
                        type=str)

    parser.add_argument("labels_key",
                        help="The key for the labels file to be changed.",
                        type=str)

    args = parser.parse_args()

    h5_path = os.path.abspath(args.input_h5)

    if not os.path.isfile(h5_path):
        print("The specified input file {} does not exist.".format(h5_path))
        sys.exit(1)

    change_dense_to_one_hot(h5_path, args.labels_key)
