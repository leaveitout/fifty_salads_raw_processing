#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import cv2
import os
import glob
import sys
import numpy as np
import progressbar


def load_depth(depth_file: os.path) -> np.ndarray:
    return cv2.imread(depth_file)
    # return cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)


def save_depth(depth_file: os.path, depth_mat: np.ndarray):
    return cv2.imwrite(depth_file, depth_mat)


def denoise_depth(depth_images: list, temporal_window_size=7) -> np.ndarray:
    return cv2.fastNlMeansDenoisingMulti(
        depth_images,
        imgToDenoiseIndex=3,
        temporalWindowSize=temporal_window_size,
        dst=None,
        templateWindowSize=7,
        searchWindowSize=21,
        # normType=NORM_L1
    )


def filter_dir(sample_dir: os.path):
    depth_dir = os.path.join(sample_dir, 'depth')

    print("Processing depth dir: {}".format(depth_dir))

    depth_files = glob.glob1(depth_dir, '*.png')
    depth_files.sort()

    depth_images = []

    temporal_window_size = 7

    # Populate the first necessary amount of files - 1
    for depth_idx in range(temporal_window_size - 1):
        current_depth_filename = depth_files[depth_idx]
        current_depth_path = os.path.join(depth_dir, current_depth_filename)
        current_depth = load_depth(current_depth_path)
        depth_images.append(current_depth)

    output_dir = os.path.join(sample_dir, 'depth_filtered')

    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    bar = progressbar.ProgressBar()
    for depth_idx in bar(range(temporal_window_size, len(depth_files))):
        current_depth_filename = depth_files[depth_idx]
        current_depth_path = os.path.join(depth_dir, current_depth_filename)
        current_depth = load_depth(current_depth_path)
        depth_images.append(current_depth)

        output_depth_path = os.path.join(output_dir, current_depth_filename)

        current_filtered_depth = denoise_depth(depth_images,
                                               temporal_window_size)

        save_depth(output_depth_path, current_filtered_depth)

        _ = depth_images.pop(0)


def filter_dataset(dataset_dir: os.path):
    sample_dirs = glob.glob1(dataset_dir, '[0-9][0-9]-[1-2]')
    sample_dirs.sort()

    sample_dirs = [os.path.join(dataset_dir, sd) for sd in sample_dirs]

    _ = [filter_dir(sd) for sd in sample_dirs]

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process the 50 Salads dataset to have better flow data"
    )
    parser.add_argument('dataset_dir', help='Path to the dataset directory.')

    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)

    if not os.path.isdir(dataset_dir):
        print("The specified input dir {} does not exist.".format(dataset_dir))
        sys.exit(1)

    sys.exit(filter_dataset(dataset_dir))
