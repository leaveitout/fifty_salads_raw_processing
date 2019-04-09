#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import cv2
import glob
import h5py
import os
import sys
import subprocess

import progressbar

import welford

import numpy as np


def load_image(image_loc):
    # load image, scale and return it as numpy array
    image = cv2.imread(image_loc)
    # image = cv2.resize(image, )
    # TODO: To remove, bug is in process.py that leaves last file full size
    # if image.shape[0] != 240:
    #     image = image[::2, ::2, ::]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image_mat: np.ndarray,
                 downscale: float,
                 cv2_interpolation=cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(image_mat,
                      dsize=None,
                      fx=downscale,
                      fy=downscale,
                      interpolation=cv2_interpolation)


def from_nhwc_to_nchw(input_data):
    """
    Converts a numpy array from xyzhwc to xyzchw.

    :param input_data: Input array with channels last.
    :return: Output array with channels third from the end.
    """
    return np.rollaxis(input_data, -1, -3)


def extract_norm_images_for_sample(sample_dir):
    pcd_dir = os.path.join(sample_dir, 'pcd')
    all_pcds = glob.glob1(pcd_dir, '*.pcd')
    all_pcds.sort()

    norm_dir = os.path.join(sample_dir, 'norm')

    if os.path.isdir(norm_dir):
        print("{} already exists, skipping".format(norm_dir))
    else:
        os.mkdir(norm_dir)

        bar = progressbar.ProgressBar()
        print("Extracting {}".format(pcd_dir))
        for pcd in bar(all_pcds):
            pcd_id = pcd.split('.')[0]
            norm_filename = pcd_id + '.png'
            norm_file_path = os.path.join(norm_dir, norm_filename)
            pcd_file_path = os.path.join(pcd_dir, pcd)
            extract_norm_cmd = ['pcl_pcd2png',
                                '--no-nan',
                                '--field',
                                'normal',
                                pcd_file_path,
                                norm_file_path]

            subprocess.call(extract_norm_cmd,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL)

    pcd_f_dir = os.path.join(sample_dir, 'pcd_f')
    all_f_pcds = glob.glob1(pcd_f_dir, '*.pcd')
    all_f_pcds.sort()
    norm_f_dir = os.path.join(sample_dir, 'norm_f')

    if os.path.isdir(norm_f_dir):
        print("{} already exists, skipping")
    else:
        os.mkdir(norm_f_dir)

        bar = progressbar.ProgressBar()
        print("Extracting {}".format(pcd_f_dir))
        for pcd in bar(all_f_pcds):
            pcd_id = pcd.split('.')[0]
            norm_filename = pcd_id + '.png'
            norm_file_path = os.path.join(norm_f_dir, norm_filename)
            pcd_file_path = os.path.join(pcd_f_dir, pcd)
            extract_norm_cmd = ['pcl_pcd2png',
                                '--no-nan',
                                '--field',
                                'normal',
                                pcd_file_path,
                                norm_file_path]

            subprocess.call(extract_norm_cmd,
                            stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL)


def add_mean_and_var(input_file_location: os.path) -> None:
    means = []
    vars = []

    with h5py.File(input_file_location, 'r+') as h5_file:
        splits = h5_file['splits'][:]
        offsets = h5_file['offsets'][:]
        old_means = h5_file['mean'][:]
        old_vars = h5_file['var'][:]
        norm = h5_file['norm']
        # norm_f = h5_file['norm_f']
        num_splits = len(splits)

        for split_idx in range(num_splits):
            print("Split {} of {}:".format(split_idx + 1, num_splits))
            sample_include = splits[split_idx]
            train_idxs = []

            for idx, inc in enumerate(sample_include):
                sample_idxs = list(range(offsets[idx], offsets[idx + 1]))

                if inc:
                    train_idxs += sample_idxs

            norm_accum = welford.WelfordAccumulator(min_valid_samples=100)
            norm_f_accum = welford.WelfordAccumulator(min_valid_samples=100)

            bar = progressbar.ProgressBar()
            for train_idx in bar(train_idxs):
                norm_image = norm[train_idx]
                # norm_f_image = norm_f[train_idx]

                # The entire dataset is covered after two splits.
                if split_idx < 2:
                    norm_image = np.nan_to_num(norm_image, copy=False)
                    norm[train_idx] = norm_image
                    norm_f_image = np.nan_to_num(norm_f_image, copy=False)
                    # norm_f[train_idx] = norm_f_image
                    # flow_image = np.nan_to_num(flow_image)
                    # flow[train_idx] = flow_image

                norm_accum.add(norm_image)
                norm_f_accum.add(norm_f_image)

            norm_mean = norm_accum.get_mean().astype(np.float32)
            norm_var = norm_accum.get_variance().astype(np.float32)
            norm_f_mean = norm_f_accum.get_mean().astype(np.float32)
            norm_f_var = norm_f_accum.get_variance().astype(np.float32)

            current_mean = old_means[split_idx]
            split_mean = np.concatenate((current_mean,
                                         norm_mean,
                                         norm_f_mean))

            current_var = old_vars[split_idx]
            split_var = np.concatenate((current_var,
                                        norm_var,
                                        norm_f_var))

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
                   norm_dataset: h5py.Dataset,
                   # norm_f_dataset: h5py.Dataset,
                   offset: int,
                   downscale: float):
    norm_dir = os.path.join(sample_dir, 'norm')
    norm_f_dir = os.path.join(sample_dir, 'norm_f')

    norm_files = glob.glob1(norm_dir, '*.png')
    norm_files.sort()
    # norm_f_files = glob.glob1(norm_f_dir, '*.png')
    # norm_f_files.sort()

    print("Adding {} norm...".format(sample_dir))

    for idx, norm_file in enumerate(norm_files):
        norm_file_path = os.path.join(norm_dir, norm_file)
        norm_mat = load_image(norm_file_path)

        if 0.0 < downscale < 1.0:
            norm_mat = resize_image(norm_mat, downscale, cv2.INTER_NEAREST)

        norm_mat = from_nhwc_to_nchw(norm_mat)
        norm_dataset[offset + idx] = norm_mat

    # print("Adding {} norm_f...".format(sample_dir))
    #
    # for idx, norm_file in enumerate(norm_f_files):
    #     norm_file_path = os.path.join(norm_f_dir, norm_file)
    #     norm_mat = load_image(norm_file_path)
    #
    #     if 0.0 < downscale < 1.0:
    #         norm_mat = resize_image(norm_mat, downscale, cv2.INTER_NEAREST)
    #
    #     norm_mat = from_nhwc_to_nchw(norm_mat)
    #     norm_f_dataset[offset + idx] = norm_mat


def add_norms_to_dataset(h5_dataset_path: os.path,
                         sample_dirs: list,
                         downscale: float):
    with h5py.File(h5_dataset_path, 'r+') as h5_file:
        norm_dataset_shape = h5_file['rgb'].shape
        norm_chunk_shape = (1,) + norm_dataset_shape[1:]

        norm_dataset = h5_file.create_dataset(name='norm',
                                              shape=norm_dataset_shape,
                                              dtype=np.uint8,
                                              compression='lzf',
                                              chunks=norm_chunk_shape)

        # norm_f_dataset = h5_file.create_dataset(name='norm_f',
        #                                         shape=norm_dataset_shape,
        #                                         dtype=np.uint8,
        #                                         compression='lzf',
        #                                         chunks=norm_chunk_shape)

        offsets = h5_file['offsets'][:]

        for sample_dir, offset in zip(sample_dirs, offsets):
            process_sample(
                sample_dir,
                norm_dataset,
                # norm_f_dataset,
                offset,
                downscale
            )


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

    # Extract the png files for each sample
    for sd in sample_dirs:
        extract_norm_images_for_sample(sd)

    add_norms_to_dataset(h5_dataset_path, sample_dirs, downscale)

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
