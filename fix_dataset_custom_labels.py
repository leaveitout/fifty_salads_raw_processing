#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import glob
import numpy as np
import h5py

mid_dict = {
    "add_dressing": 0,
    "add_oil": 1,
    "add_pepper": 2,
    "add_salt": 3,
    "add_vinegar": 4,
    "background": 5,
    "cut_cheese": 6,
    "cut_cucumber": 7,
    "cut_lettuce": 8,
    "cut_tomato": 9,
    "mix_dressing": 10,
    "mix_ingredients": 11,
    "peel_cucumber": 12,
    "place_cheese_into_bowl": 13,
    "place_cucumber_into_bowl": 14,
    "place_lettuce_into_bowl": 15,
    "place_tomato_into_bowl": 16,
    "serve_salad_onto_plate": 17
}

custom_dict = {
    "add_dressing": 0,
    "add_oil": 1,
    "add_pepper": 2,
    "background": 3,
    "cut_into_pieces": 4,
    "mix_dressing": 5,
    "mix_ingredients": 6,
    "peel_cucumber": 7,
    "place_into_bowl": 8,
    "serve_salad_onto_plate": 9
}

mid_to_custom = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 4,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 8,
    14: 8,
    15: 8,
    16: 8,
    17: 9
}


def fix_sample_dir_labels(sample_dir: os.path):
    mid_labels = np.load(os.path.join(sample_dir, 'labels_mid.npy'))

    custom_labels = np.zeros((mid_labels.shape[0], 10), dtype=np.float)

    for idx in np.arange(mid_labels.shape[0]):
        mid_label_id = np.argmax(mid_labels[idx])
        custom_labels[idx, mid_to_custom[mid_label_id]] = 1.0

    np.save('labels_custom.npy', custom_labels)


def fix_labels(dataset_loc: os.path):
    sample_dirs = glob.glob1(dataset_dir, '[0-9][0-9]-[1-2]')
    sample_dirs.sort()

    sample_dirs = [os.path.join(dataset_dir, sd) for sd in sample_dirs]

    _ = [fix_sample_dir_labels(sd) for sd in sample_dirs]

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate and add the mean and var for each split")
    parser.add_argument("input", help="The input dataset dir.", type=str)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.input)

    if not os.path.isdir(dataset_dir):
        print("The specified dataset dir {} "
              "does not exist.".format(dataset_dir))
        sys.exit(1)

    try:
        fix_labels(dataset_dir)
    except ValueError as e:
        print("Error encountered: {}".format(e))
        sys.exit(1)
