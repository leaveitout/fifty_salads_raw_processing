#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys

import h5py
import progressbar
import numpy as np


def copy_labels_splits_offsets(input_h5_loc: os.path, output_h5_loc: os.path):
    labels_keys = ['mid', 'splits', 'offsets']

    for k in labels_keys:
        with h5py.File(input_h5_loc, 'r') as input_h5:
            label_dataset = input_h5[k]

            print("Extracting {} dataset.".format(k))
            with h5py.File(output_h5_loc, 'a') as output_h5:
                output_h5.create_dataset(k, data=label_dataset)


def create_dataset(input_h5_loc: os.path,
                   output_h5_loc: os.path,
                   dataset_key: str,
                   start_x_crop: int,
                   stop_x_crop: int,
                   start_y_crop: int,
                   stop_y_crop: int):
    with h5py.File(input_h5_loc, 'r') as input_h5:
        input_dataset = input_h5[dataset_key]
        with h5py.File(output_h5_loc, 'a') as output_h5:
            output_dataset_chunk_shape = (
                1,
                3,
                stop_y_crop - start_y_crop,
                stop_x_crop - start_x_crop
            )

            output_dataset_shape = (
                input_dataset.shape[0],
                3,
                output_dataset_chunk_shape[2],
                output_dataset_chunk_shape[3]
            )

            output_dataset = output_h5.create_dataset(
                dataset_key,
                shape=output_dataset_shape,
                dtype=input_dataset.dtype,
                chunks=output_dataset_chunk_shape,
                compression='lzf'
            )

            bar = progressbar.ProgressBar()
            print("Extracting {} dataset".format(dataset_key))
            for idx in bar(range(input_dataset.shape[0])):
                output_dataset[idx] = input_dataset[
                                      idx,
                                      :,
                                      start_y_crop: stop_y_crop,
                                      start_x_crop: stop_x_crop]


def create_rgbflow_datasets(input_h5_loc: os.path,
                            output_h5_loc: os.path,
                            start_x_crop: int,
                            stop_x_crop: int,
                            start_y_crop: int,
                            stop_y_crop: int,
                            stride: int):
    rgbflow_key = 'rgb'
    with h5py.File(input_h5_loc, 'r') as input_h5:
        input_rgbrflow_dataset = input_h5[rgbflow_key]

        with h5py.File(output_h5_loc, 'a') as output_h5:
            output_chunk_shape = (
                1,
                6,
                stop_y_crop - start_y_crop,
                stop_x_crop - start_x_crop
            )

            output_datasets = []

            num_samples = input_rgbrflow_dataset.shape[0]

            num_samples_per_stride = [
                num_samples // stride + int((num_samples % stride) > idx)
                for idx in range(stride)
            ]

            for stride_idx in range(stride):
                output_shape = (
                    num_samples_per_stride[stride_idx],
                    6,
                    stop_y_crop - start_y_crop,
                    stop_x_crop - start_x_crop
                )

                output_dataset = output_h5.create_dataset(
                    'rgbflow' + str(stride_idx),
                    shape=output_shape,
                    dtype=np.uint8,
                    chunks=output_chunk_shape,
                    compression='lzf'
                )

                output_datasets.append(output_dataset)

            print("Extracting rgbflow dataset")
            bar = progressbar.ProgressBar()
            for sample_idx in bar(range(num_samples)):
                rgb_flow = input_rgbrflow_dataset[
                      sample_idx,
                      :,
                      start_y_crop: stop_y_crop,
                      start_x_crop: stop_x_crop
                      ]

                dataset_idx = sample_idx % stride

                output_datasets[dataset_idx][sample_idx // stride] = rgb_flow


def create_stats_datasets(input_h5_loc: os.path,
                          output_h5_loc: os.path,
                          start_x_crop: int,
                          stop_x_crop: int,
                          start_y_crop: int,
                          stop_y_crop: int):
    mean_key = 'mean'
    var_key = 'var'

    image_type_mean_planes = {
        'rgb': (0, 1, 2),
        'depth': (3, 4, 5),
        'norm': (6, 7, 8),
        'flow': (9, 10, 11)
    }

    with h5py.File(input_h5_loc, 'r') as input_h5:
        input_mean_dataset = input_h5[mean_key]
        input_var_dataset = input_h5[var_key]

        # Read the mean and var dataset
        output_mean = input_mean_dataset[
                      :,
                      start_y_crop: stop_y_crop,
                      start_x_crop: stop_x_crop
                      ]

        output_var = input_var_dataset[
                     :,
                     start_y_crop: stop_y_crop,
                     start_x_crop: stop_x_crop
                     ]

        with h5py.File(output_h5_loc, 'a') as output_h5:
            print("Extracting mean dataset")
            output_h5.create_dataset('mean', data=output_mean)

            print("Extracting var dataset")
            output_h5.create_dataset('var', data=output_var)


def create_strided_dataset(input_h5_loc: os.path,
                           output_h5_loc: os.path,
                           start_x_crop: int,
                           stop_x_crop: int,
                           start_y_crop: int,
                           stop_y_crop: int,
                           stride: int):
    copy_labels_splits_offsets(input_h5_loc, output_h5_loc)

    create_rgbflow_datasets(
        input_h5_loc,
        output_h5_loc,
        start_x_crop,
        stop_x_crop,
        start_y_crop,
        stop_y_crop,
        stride
    )

    create_dataset(
        input_h5_loc,
        output_h5_loc,
        'depth',
        start_x_crop,
        stop_x_crop,
        start_y_crop,
        stop_y_crop,
    )

    create_stats_datasets(
        input_h5_loc,
        output_h5_loc,
        start_x_crop,
        stop_x_crop,
        start_y_crop,
        stop_y_crop,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Make a strided copy of an OSCE dataset."
    )

    parser.add_argument("input",
                        help="The input h5 file for the dataset.",
                        type=str)

    parser.add_argument("output",
                        help="The output h5 file to be created.",
                        type=str)

    parser.add_argument("startx",
                        help="The start of the x crop.",
                        type=int)

    parser.add_argument("stopx",
                        help="The stop of the x crop.",
                        type=int)

    parser.add_argument("starty",
                        help="The start of the y crop.",
                        type=int)

    parser.add_argument("stopy",
                        help="The stop of the y crop.",
                        type=int)

    parser.add_argument("stride",
                        help="The striding to use.",
                        type=int)

    args = parser.parse_args()

    if os.path.isfile(args.input):
        create_strided_dataset(
            args.input,
            args.output,
            args.startx,
            args.stopx,
            args.starty,
            args.stopy,
            args.stride
        )
    else:
        print("File does not exist at location, exiting...")
        sys.exit(1)

