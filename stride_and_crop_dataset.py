#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys

import h5py
import progressbar
import numpy as np


def copy_labels_splits_offsets(input_h5_loc: os.path, output_h5_loc: os.path):
    labels_keys = ['coarse', 'mid', 'fine', 'custom', 'splits', 'offsets']

    for k in labels_keys:
        with h5py.File(input_h5_loc, 'r') as input_h5:
            label_dataset = input_h5[k]

            print("Extracting {} dataset.".format(k))
            with h5py.File(output_h5_loc, 'a') as output_h5:
                output_h5.create_dataset(k, data=label_dataset)


def create_depth_dataset(input_h5_loc: os.path,
                         output_h5_loc: os.path,
                         start_x_crop: int,
                         stop_x_crop: int,
                         start_y_crop: int,
                         stop_y_crop: int):
    depth_key = 'depth_f'
    with h5py.File(input_h5_loc, 'r') as input_h5:
        input_depth_dataset = input_h5[depth_key]
        with h5py.File(output_h5_loc, 'a') as output_h5:
            output_depth_chunk_shape = (
                1,
                1,
                stop_y_crop - start_y_crop,
                stop_x_crop - start_x_crop
            )

            output_depth_dataset_shape = (
                input_depth_dataset.shape[0],
                1,
                output_depth_chunk_shape[2],
                output_depth_chunk_shape[3]
            )

            output_depth_dataset = output_h5.create_dataset(
                'depth',
                shape=output_depth_dataset_shape,
                dtype=input_depth_dataset.dtype,
                chunks=output_depth_chunk_shape,
                compression='lzf'
            )

            bar = progressbar.ProgressBar()
            print("Extracting depth dataset")
            for idx in bar(range(input_depth_dataset.shape[0])):
                output_depth_dataset[idx] = input_depth_dataset[
                                            idx,
                                            :,
                                            start_y_crop: stop_y_crop,
                                            start_x_crop: stop_x_crop
                                            ]


def create_rgbflow_datasets(input_h5_loc: os.path,
                            output_h5_loc: os.path,
                            start_x_crop: int,
                            stop_x_crop: int,
                            start_y_crop: int,
                            stop_y_crop: int,
                            stride: int):
    rgb_key = 'rgb'
    flow_key = 'flowuint8'
    with h5py.File(input_h5_loc, 'r') as input_h5:
        input_rgb_dataset = input_h5[rgb_key]
        input_flow_dataset = input_h5[flow_key]

        with h5py.File(output_h5_loc, 'a') as output_h5:
            output_chunk_shape = (
                1,
                6,
                stop_y_crop - start_y_crop,
                stop_x_crop - start_x_crop
            )

            output_datasets = []

            num_samples = input_rgb_dataset.shape[0]

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
                rgb = input_rgb_dataset[
                      sample_idx,
                      :,
                      start_y_crop: stop_y_crop,
                      start_x_crop: stop_x_crop
                      ]
                flow = input_flow_dataset[
                       sample_idx,
                       :,
                       start_y_crop: stop_y_crop,
                       start_x_crop: stop_x_crop
                       ]
                rgb_flow = np.concatenate((rgb, flow), axis=-3)

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
        'depth': (3,),
        'depth_f': (4,),
        'flow': (5, 6, 7),
        'norm': (8, 9, 10),
        'normf': (11, 12, 13),
        'curvature': (14,)
    }

    with h5py.File(input_h5_loc, 'r') as input_h5:
        input_mean_dataset = input_h5[mean_key]
        input_var_dataset = input_h5[var_key]

        # Read the mean and var dataset
        rgb_means = input_mean_dataset[
                    :,
                    image_type_mean_planes['rgb'],
                    start_y_crop: stop_y_crop,
                    start_x_crop: stop_x_crop
                    ]

        depth_means = input_mean_dataset[
                      :,
                      image_type_mean_planes['depth_f'],
                      start_y_crop: stop_y_crop,
                      start_x_crop: stop_x_crop
                      ]

        flow_means = np.zeros_like(rgb_means)

        rgb_vars = input_var_dataset[
                   :,
                   image_type_mean_planes['rgb'],
                   start_y_crop: stop_y_crop,
                   start_x_crop: stop_x_crop
                   ]

        depth_vars = input_var_dataset[
                     :,
                     image_type_mean_planes['depth_f'],
                     start_y_crop: stop_y_crop,
                     start_x_crop: stop_x_crop
                     ]

        flow_vars = 255.0 * np.ones_like(rgb_vars)

        output_means = np.concatenate((rgb_means,
                                       depth_means,
                                       rgb_means,
                                       flow_means),
                                      axis=-3)

        output_vars = np.concatenate((rgb_vars,
                                      depth_vars,
                                      rgb_vars,
                                      flow_vars),
                                     axis=-3)

        with h5py.File(output_h5_loc, 'a') as output_h5:
            print("Extracting mean dataset")
            output_h5.create_dataset('mean', data=output_means)

            print("Extracting var dataset")
            output_h5.create_dataset('var', data=output_vars)


def create_strided_dataset(input_h5_loc: os.path,
                           output_h5_loc: os.path,
                           start_x_crop: int,
                           stop_x_crop: int,
                           start_y_crop: int,
                           stop_y_crop: int,
                           stride: int):
    copy_labels_splits_offsets(input_h5_loc, output_h5_loc)

    create_depth_dataset(
        input_h5_loc,
        output_h5_loc,
        start_x_crop,
        stop_x_crop,
        start_y_crop,
        stop_y_crop,
    )

    create_rgbflow_datasets(
        input_h5_loc,
        output_h5_loc,
        start_x_crop,
        stop_x_crop,
        start_y_crop,
        stop_y_crop,
        stride
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
        description="Add mid and custom granularities to a dataset, "
                    "based on the existing fine granularity."
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
