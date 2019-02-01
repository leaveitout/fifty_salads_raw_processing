#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2018.
Contact sbruton[á]tcd.ie.
"""
import argparse
import cv2
import glob
import h5py
import os
import sys
import subprocess

from pypcd.pypcd import PointCloud

import progressbar

import welford

import numpy as np




def load_curvature(cloud_loc: os.path):
    cloud = PointCloud.from_path(cloud_loc)
    curvature = np.nan_to_num(cloud.pc_data['curvature'], copy=False)

    if len(curvature.shape) == 2:
        curvature = np.reshape(curvature, curvature.shape + (1,))

    return curvature


def resize_curvature(image_mat: np.ndarray,
                     downscale: float) -> np.ndarray:
    step = int(1 // downscale)
    return image_mat[::step, ::step, :]


def from_nhwc_to_nchw(input_data):
    """
    Converts a numpy array from xyzhwc to xyzchw.

    :param input_data: Input array with channels last.
    :return: Output array with channels third from the end.
    """
    return np.rollaxis(input_data, -1, -3)


def add_mean_and_var(input_file_location: os.path) -> None:
    means = []
    vars = []

    with h5py.File(input_file_location, 'r+') as h5_file:
        splits = h5_file['splits'][:]
        offsets = h5_file['offsets'][:]
        old_means = h5_file['mean'][:]
        old_vars = h5_file['var'][:]
        norm = h5_file['norm']
        curvature = h5_file['curvature']
        num_splits = len(splits)

        for split_idx in range(num_splits):
            print("Split {} of {}:".format(split_idx + 1, num_splits))
            sample_include = splits[split_idx]
            train_idxs = []

            for idx, inc in enumerate(sample_include):
                sample_idxs = list(range(offsets[idx], offsets[idx + 1]))

                if inc:
                    train_idxs += sample_idxs

            curvature_accum = welford.WelfordAccumulator(min_valid_samples=100)

            bar = progressbar.ProgressBar()
            for train_idx in bar(train_idxs):
                curvature_image = curvature[train_idx]

                # The entire dataset is covered after two splits.
                if split_idx < 2:
                    curvature_image = np.nan_to_num(curvature_image, copy=False)
                    curvature[train_idx] = curvature_image

                curvature_accum.add(curvature_image)

            curvature_mean = curvature_accum.get_mean().astype(np.float32)
            curvature_var = curvature_accum.get_variance().astype(np.float32)

            current_mean = old_means[split_idx]
            split_mean = np.concatenate((current_mean,
                                         curvature_mean))

            current_var = old_vars[split_idx]
            split_var = np.concatenate((current_var,
                                        curvature_var))

            means.append(split_mean)
            vars.append(split_var)

        means = np.array(means)
        vars = np.array(vars)

        print(means.shape)
        print(vars.shape)
        print(means[0])
        print(vars[0])

        del h5_file["mean"]

        h5_file.create_dataset(
            name="mean",
            shape=means.shape,
            dtype=means.dtype,
            data=means
        )

        del h5_file["var"]
        h5_file.create_dataset(
            name="var",
            shape=vars.shape,
            dtype=vars.dtype,
            data=vars
        )


def check_files_correct(rgb_files, norm_files, sample_dir):
    # Sense check the files
    for rgb_file, norm_file in zip(rgb_files, norm_files):
        if rgb_file.split('.')[0] != norm_file.split('.')[0]:
            raise ValueError("{} does not match {} "
                             "for sample dir {}".format(rgb_file,
                                                        norm_file,
                                                        sample_dir))

    if len(rgb_files) != len(norm_files):
        raise ValueError("Sample dir {} does not contain equal "
                         "number of rgb and depth files".format(sample_dir))


def process_sample(sample_dir: os.path,
                   curvature_dataset: h5py.Dataset,
                   offset: int,
                   downscale: float):
    curvature_dir = os.path.join(sample_dir, 'pcd')

    curvature_files = glob.glob1(curvature_dir, '*.pcd')
    curvature_files.sort()

    print("Adding {} curvature...".format(sample_dir))

    for idx, curvature_file in enumerate(curvature_files):
        curvature_file_path = os.path.join(curvature_dir, curvature_file)
        curvature_mat = load_curvature(curvature_file_path)

        if 0.0 < downscale < 1.0:
            curvature_mat = resize_curvature(curvature_mat, downscale)

        curvature_mat = from_nhwc_to_nchw(curvature_mat)
        curvature_dataset[offset + idx] = curvature_mat


def add_curvature_to_dataset(h5_dataset_path: os.path,
                             sample_dirs: list,
                             downscale: float):
    with h5py.File(h5_dataset_path, 'r+') as h5_file:
        curvature_dataset_shape = h5_file['depth'].shape
        curvature_chunk_shape = (1,) + curvature_dataset_shape[1:]

        curvature_dataset = h5_file.create_dataset(
            name='curvature',
            shape=curvature_dataset_shape,
            dtype=np.float32,
            compression='lzf',
            chunks=curvature_chunk_shape
        )

        offsets = h5_file['offsets'][:]

        for sample_dir, offset in zip(sample_dirs, offsets):
            process_sample(sample_dir, curvature_dataset, offset, downscale)


def process_dataset(h5_dataset_path: os.path,
                    dataset_path: os.path,
                    downscale: float):
    # TODO: Modify the dataset to include these statistics
    sample_dirs = glob.glob1(dataset_path, "[0-9][0-9]-[0-9]")
    if len(sample_dirs) != 50:
        raise ValueError("There should be 50 sample folders, "
                         "found {}.".format(len(sample_dirs)))

    sample_dirs.sort()
    sample_dirs = [os.path.join(dataset_path, s) for s in sample_dirs]

    add_curvature_to_dataset(h5_dataset_path, sample_dirs, downscale)

    # Calculate the mean and std norms
    add_mean_and_var(h5_dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a h5 dataset from a 50Salads preprocessed folder.")
    parser.add_argument("input",
                        help="The directory containing the processed samples.",
                        type=str)
    parser.add_argument("output",
                        help="The .h5 dataset to output",
                        type=str)
    parser.add_argument("--downscale",
                        help="The downscale factor.",
                        default=1.0,
                        type=float)
    args = parser.parse_args()

    try:
        dataset_path = os.path.abspath(args.input)
        output_h5_path = os.path.abspath(args.output)

        if not os.path.isdir(dataset_path):
            raise ValueError("Invalid directory for h5 dataset.")

        if not os.path.isfile(output_h5_path):
            raise ValueError("Invalid h5 dataset.")

        process_dataset(h5_dataset_path=output_h5_path,
                        dataset_path=dataset_path,
                        downscale=args.downscale)

    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
