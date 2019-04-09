#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import argparse
import h5py
import os
import sys
import numpy as np
import progressbar


def add_mean_to_invalid_values(input_file_location: os.path) -> None:
    with h5py.File(input_file_location, 'r+') as h5_file:
        # Get image shape
        depth = h5_file['depth_jet']
        norm = h5_file['norm']
        num_samples = norm.shape[0]
        mean_images = np.moveaxis(h5_file['mean'][:], 0, -1)
        norm_mean = np.squeeze(mean_images[0, 0, 6:9]).astype(np.uint8)
        depth_mean = np.squeeze(mean_images[0, 0, 3:6]).astype(np.uint8)

        print("Processing depth images")
        bar = progressbar.ProgressBar()
        for image_idx in bar(range(num_samples)):
            depth_image = depth[image_idx]
            depth_image = np.moveaxis(depth_image, 0, -1)

            depth_image[
                (depth_image[:, :, 0] == 72) &
                (depth_image[:, :, 1] == 72) &
                (depth_image[:, :, 2] == 72)
            ] = depth_mean

            depth_image = np.moveaxis(depth_image, -1, 0)
            depth[image_idx] = depth_image

        print("Processing norm images")
        bar = progressbar.ProgressBar()
        for image_idx in bar(range(num_samples)):
            norm_image = norm[image_idx]
            norm_image = np.moveaxis(norm_image, 0, -1)

            norm_image[
                (norm_image[:, :, 0] == 89) &
                (norm_image[:, :, 1] == 89) &
                (norm_image[:, :, 2] == 89)
            ] = norm_mean

            norm_image = np.moveaxis(norm_image, -1, 0)
            norm[image_idx] = norm_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate mean and var single values for each image type")
    parser.add_argument("input", help="The input h5 file.", type=str)
    args = parser.parse_args()

    input_loc = os.path.abspath(args.input)

    if not os.path.isfile(input_loc):
        print("The specified input file {} does not exist.".format(input_loc))
        sys.exit(1)

    try:
        add_mean_to_invalid_values(input_loc)
    except ValueError as e:
        print("Error encountered: {}".format(e))
        sys.exit(1)
