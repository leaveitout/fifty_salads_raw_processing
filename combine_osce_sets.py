#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import glob
import json
from functools import partial

import numpy as np
import cv2
import h5py
import progressbar


def colormap_depth_image(input_depth_image_uint8: np.ndarray) -> np.ndarray:
    input_depth_image_uint8 = np.squeeze(input_depth_image_uint8)
    mapped = cv2.applyColorMap(input_depth_image_uint8, cv2.COLORMAP_HOT)
    mapped = np.moveaxis(mapped, -1, 0)
    return mapped


def get_offsets(samples_dir: os.path) -> np.ndarray:
    h5_filenames = glob.glob1(samples_dir, '*.h5')
    h5_filenames.sort()

    offsets = [0]
    total = 0
    for f in h5_filenames:
        with h5py.File(os.path.join(samples_dir, f)) as sample_h5:
            total += sample_h5['rgbd'].shape[0]
            offsets.append(total)

    return np.array(offsets)


def get_splits(offsets: np.ndarray, num_splits: int) -> np.ndarray:
    """
    Generate a numpy array that indicates inclusion for each sample per split.
    Note: This function assumes that there is the number of samples
    is divided evenly by the number of offsets.

    :rtype: The boolean numpy array indicating inclusion.
    """
    if (offsets.shape[0] - 1) % num_splits:
        raise ValueError("The no. of splits should divide the no. of samples")

    splits = np.ones((num_splits, offsets.shape[0] - 1), dtype=np.bool)
    samples_per_split = splits.shape[1] // splits.shape[0]

    for row in range(splits.shape[0]):
        test_sample_num = splits.shape[0] - 1 - row
        test_sample_start = samples_per_split * test_sample_num
        splits[row,
        test_sample_start: test_sample_start + samples_per_split] = False

    return splits


def get_mid_labels(dataset_dir: os.path, offsets: np.ndarray) -> np.ndarray:
    # Get the list of subdirectories, sorted
    # recording_sample_dirs = [
    #     os.path.join(dataset_dir, s)
    #     for s in os.listdir(dataset_dir)
    #     if os.path.isdir(os.path.join(dataset_dir, s))
    # ]
    # recording_sample_dirs.sort()

    # sample_label_files = [
    #     os.path.join(rsd, 'labels.json')
    #     for rsd in recording_sample_dirs
    # ]
    all_labels = np.empty(offsets[-1], dtype=np.int32)

    sample_label_files = [
        os.path.join(dataset_dir, s, 'labels.json')
        for s in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, s))
    ]
    sample_label_files.sort()

    total = 0
    for slf in sample_label_files:
        with open(slf, 'r') as json_file:
            labels_data = json.load(json_file)['labels']
            labels = [ld['action'] for ld in labels_data]
            all_labels[total: total + len(labels)] = labels
            total += len(labels)

    if total != offsets[-1]:
        raise ValueError("No. labels {} != No. samples")

    return all_labels


def convert_flow_image(
        input_flow_image: np.ndarray,
        min_flow,
        max_flow,
        scale
):
    rescaled_flow_image = np.clip(input_flow_image, min_flow, max_flow)
    rescaled_flow_image += min_flow
    rescaled_flow_image *= scale
    return rescaled_flow_image.astype(np.uint8)


def center_holes(
        depth_image: np.ndarray,
        mean_value: int
) -> np.ndarray:
    depth_image[np.where(depth_image == 0)] = mean_value
    return depth_image


def get_means(image_size: tuple) -> np.ndarray:
    rgb_mean = np.full(
        shape=(3,) + image_size,
        fill_value=127.5,
        dtype=np.float32
    )

    # Make a depth image and then convert it
    depth_mean = np.full(image_size, fill_value=128, dtype=np.uint8)
    depth_mean = colormap_depth_image(depth_mean)

    norm_mean = np.full_like(rgb_mean, fill_value=0.0)

    flow_mean = np.full_like(rgb_mean, fill_value=129.0)

    return np.concatenate((rgb_mean, depth_mean, norm_mean, flow_mean))


def get_variances(image_size: tuple) -> np.ndarray:
    rgb_var = np.full(
        shape=(3,) + image_size,
        fill_value=127.5,
        dtype=np.float32
    )

    # Make a depth image and then convert it
    depth_zeros_uint8 = np.full(image_size, fill_value=0, dtype=np.uint8)
    depth_ones_uint8 = np.full(image_size, fill_value=255, dtype=np.uint8)
    depth_zeros = colormap_depth_image(depth_zeros_uint8)
    depth_ones = colormap_depth_image(depth_ones_uint8)
    depth_sample = np.stack((depth_zeros, depth_ones))
    depth_var = np.var(depth_sample, axis=0)

    norm_var = np.full_like(rgb_var, fill_value=1.0)

    flow_var = np.full_like(rgb_var, fill_value=129.0)

    return np.concatenate((rgb_var, depth_var, norm_var, flow_var))


def add_images(
        samples_dir: os.path,
        image_shape: tuple,
        output_h5_path: os.path,
        offsets: np.ndarray
) -> None:
    num_frames = offsets[-1]
    rgbflow_dataset_shape = (num_frames, 6,) + image_shape
    rgbflow_chunk_shape = (1, 6,) + image_shape
    depth_dataset_shape = (num_frames, 3,) + image_shape
    depth_chunk_shape = (1, 3,) + image_shape
    with h5py.File(output_h5_path, 'r+') as h5_file:
        rgbflow_dataset = h5_file.create_dataset(name='rgbflow',
                                                 shape=rgbflow_dataset_shape,
                                                 dtype=np.uint8,
                                                 compression='lzf',
                                                 chunks=rgbflow_chunk_shape)

        depth_dataset = h5_file.create_dataset(name='depth',
                                               shape=depth_dataset_shape,
                                               dtype=np.uint8,
                                               compression='lzf',
                                               chunks=depth_chunk_shape)

        sample_h5_filenames = glob.glob1(samples_dir, '*.h5')
        sample_h5_filenames.sort()

        # Values for conversion of flow to uint8
        max_abs_flow = 0.05
        distinct_values = 256

        flow_increments = max_abs_flow / (distinct_values // 2)
        max_flow = max_abs_flow
        min_flow = max_flow - (flow_increments * (distinct_values - 1))
        scale = (distinct_values - 1) / (max_flow - min_flow)

        flow_to_uint8 = partial(
            convert_flow_image,
            min_flow=min_flow,
            max_flow=max_flow,
            scale=scale
        )

        # Statistic for depth values
        mean_depth_value = 128

        total = 0

        for f in sample_h5_filenames:
            with h5py.File(os.path.join(samples_dir, f)) as sample_h5:
                rgbd_dataset = sample_h5['rgbd']
                flow_dataset = sample_h5['flow']

                num_sample_frames = rgbd_dataset.shape[0]

                print("Extracting sample: {}".format(f))
                bar = progressbar.ProgressBar()
                for idx in bar(range(num_sample_frames)):
                    rgb_frame, depth_frame = np.split(rgbd_dataset[idx], [3])

                    # Fill the holes with the mean depth before conversion
                    depth_frame = center_holes(depth_frame, mean_depth_value)
                    depth_frame = colormap_depth_image(depth_frame)

                    # Convert flow to uint8
                    flow_frame = flow_to_uint8(np.nan_to_num(flow_dataset[idx]))

                    rgbflow_frame = np.concatenate((rgb_frame, flow_frame))

                    rgbflow_dataset[total + idx] = rgbflow_frame
                    depth_dataset[total + idx] = depth_frame

                total += num_sample_frames


def combine_datasets(
        samples_dir: os.path,
        dataset_dir: os.path,
        output_h5_path: os.path,
        num_splits: int
):
    offsets = get_offsets(samples_dir)
    splits = get_splits(offsets, num_splits)
    mid_labels = get_mid_labels(dataset_dir, offsets)
    # TODO: Do this programmatically.
    image_size = (120, 160)
    means = get_means(image_size)
    variances = get_variances(image_size)

    # Set up the datasets within the h5file.
    with h5py.File(output_h5_path, 'w') as h5_file:
        # offsets
        h5_file.create_dataset('offsets', data=offsets)

        # splits
        h5_file.create_dataset('splits', data=splits)

        # mid granularity
        h5_file.create_dataset('mid', data=mid_labels)

        # means
        h5_file.create_dataset(name="mean", data=means)

        # vars
        h5_file.create_dataset(name="var", data=variances)

    # rgbflow & depth
    add_images(samples_dir, image_size, output_h5_path, offsets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine processed h5 sets for individual OSCE samples.")
    parser.add_argument("sample_dir",
                        help="The directory containing the processed sets.",
                        type=str)
    parser.add_argument("dataset_dir",
                        help="The directory containing the original dataset.",
                        type=str)
    parser.add_argument("output",
                        help="The output combined h5 dataset.",
                        type=str)
    parser.add_argument("num_splits",
                        help="The number of splits to use.",
                        type=int)
    args = parser.parse_args()

    try:
        samples_path = os.path.abspath(args.sample_dir)
        dataset_path = os.path.abspath(args.dataset_dir)
        output_h5_file_path = os.path.abspath(args.output)
        output_h5_parent_path = os.path.dirname(output_h5_file_path)

        if not os.path.isdir(dataset_path):
            raise ValueError("Invalid directory for h5 dataset.")
        if not os.path.isdir(output_h5_parent_path):
            raise ValueError("Invalid directory for h5 dataset.")
        elif os.path.isfile(output_h5_file_path):
            print("Output h5 dataset file already exists, overwriting.")

        combine_datasets(
            samples_path,
            dataset_path,
            output_h5_file_path,
            args.num_splits
        )

    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
