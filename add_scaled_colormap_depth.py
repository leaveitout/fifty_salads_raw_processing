#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import numpy as np
import cv2
import h5py
import progressbar


def find_max_depth(h5_dataset_path: os.path) -> int:
    with h5py.File(h5_dataset_path, 'r') as h5_file:
        depth_dataset = h5_file['depth']
        num_samples = h5_file['offsets'][1]  # Use first split only.

        print("Finding depth dataset statistics for scaling.")

        max_depth = 0
        bar = progressbar.ProgressBar()
        for k in bar(range(num_samples)):
            curr_depth_max = int(np.max(depth_dataset[k]))
            if curr_depth_max > max_depth:
                max_depth = curr_depth_max

        print("Found max depth: {}".format(max_depth))

        return max_depth


def colormap_depth_image(input_depth_image_uint8: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(input_depth_image_uint8, cv2.COLORMAP_JET)


def scale_depth_image(input_depth_image: np.ndarray, max_depth) -> np.ndarray:
    return (255.0 * (input_depth_image / max_depth)).astype(np.uint8)


def add_scaled_colormap_depth_dataset(dataset_path: os.path):
    # First find the maximum value for the color map
    max_depth = find_max_depth(dataset_path)

    with h5py.File(dataset_path, 'r+') as h5_file:
        depth_dataset = h5_file['depth']
        depth_chunk_shape = (1,) + depth_dataset.shape[1:]
        num_samples = depth_dataset.shape[0]

        depth_jet_dataset = h5_file.create_dataset(name='depth_jet',
                                                   shape=depth_dataset.shape,
                                                   dtype=np.uint8,
                                                   compression='lzf',
                                                   chunks=depth_chunk_shape)

        print("Scaling and colormapping dataset....")

        bar = progressbar.ProgressBar()
        for idx in bar(range(num_samples)):
            scaled_depth_image = scale_depth_image(depth_dataset[idx],
                                                   max_depth)
            color_depth_image = colormap_depth_image(scaled_depth_image)
            depth_jet_dataset[idx] = color_depth_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create scaled colormapped depth from short dataset.")
    parser.add_argument(
        "input",
        help="The input h5 file. It should contain a dataset called 'depth' of "
             "type uint16.",
        type=str)
    args = parser.parse_args()

    try:
        h5_file_path = os.path.abspath(args.input)

        if not os.path.isfile(h5_file_path):
            raise ValueError("Invalid h5 dataset location.")

        add_scaled_colormap_depth_dataset(h5_file_path)
    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
