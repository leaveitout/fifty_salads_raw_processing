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
import welford


def add_mean_and_var(input_file_location: os.path) -> None:
    with h5py.File(input_file_location, 'r+') as h5_file:
        # Get image shape
        splits = h5_file['splits'][:]
        offsets = h5_file['offsets'][:]
        depth = h5_file['depth_jet']
        norm = h5_file['norm']
        image_shape = h5_file['rgb'].shape[-3:]

        # RGB mean, var
        rgb_mean = 127.5 * np.ones(image_shape, dtype=np.float32)
        rgb_var = 127.5 * np.ones(image_shape, dtype=np.float32)

        # Flow mean, var, 129.0 from uint8 mapping
        flow_mean = 129.0 * np.ones(image_shape, dtype=np.float32)
        flow_var = 129.0 * np.ones(image_shape, dtype=np.float32)

        split_idx = 0
        sample_include = splits[split_idx]
        test_idxs = []

        for idx, inc in enumerate(sample_include):
            sample_idxs = list(range(offsets[idx], offsets[idx + 1]))

            if not inc:
                test_idxs += sample_idxs

        depth_accum = welford.WelfordAccumulator(min_valid_samples=100)
        norm_accum = welford.WelfordAccumulator(min_valid_samples=100)

        bar = progressbar.ProgressBar()
        for image_idx in bar(test_idxs):
            depth_image = depth[image_idx]
            norm_image = norm[image_idx]

            # Change zeros to nans
            depth_image = np.moveaxis(depth_image, 0, -1).astype(np.float32)
            depth_image[(depth_image[:, :, 0] == 72) &
                        (depth_image[:, :, 1] == 72) &
                        (depth_image[:, :, 2] == 72)] = np.nan

            norm_image = np.moveaxis(norm_image, 0, -1).astype(np.float32)
            norm_image[(norm_image[:, :, 0] == 89) &
                       (norm_image[:, :, 1] == 89) &
                       (norm_image[:, :, 2] == 89)] = np.nan

            depth_accum.add(depth_image)
            norm_accum.add(norm_image)

        depth_mean_image = depth_accum.get_mean().astype(np.float32)
        depth_var_image = depth_accum.get_variance().astype(np.float32)
        norm_mean_image = norm_accum.get_mean().astype(np.float32)
        norm_var_image = norm_accum.get_variance().astype(np.float32)

        # Remove the zero mean values and unit variance values
        # and get a single values for statistic
        depth_mean_valid_values = depth_mean_image[
            np.logical_not(
                (depth_mean_image[:, :, 0] == 0.0) &
                (depth_mean_image[:, :, 1] == 0.0) &
                (depth_mean_image[:, :, 2] == 0.0)
            )
        ]

        depth_mean_single_value = \
            np.mean(depth_mean_valid_values, axis=0).reshape(3, 1, 1)

        depth_var_valid_values = depth_var_image[
            np.logical_not(
                (depth_var_image[:, :, 0] == 1.0) &
                (depth_var_image[:, :, 1] == 1.0) &
                (depth_var_image[:, :, 2] == 1.0)
            )
        ]

        depth_var_single_value = \
            np.mean(depth_var_valid_values, axis=0).reshape(3, 1, 1)

        norm_mean_valid_values = norm_mean_image[
            np.logical_not(
                (norm_mean_image[:, :, 0] == 0.0) &
                (norm_mean_image[:, :, 1] == 0.0) &
                (norm_mean_image[:, :, 2] == 0.0)
            )
        ]

        norm_mean_single_value = \
            np.mean(norm_mean_valid_values, axis=0).reshape(3, 1, 1)

        norm_var_valid_values = norm_var_image[
            np.logical_not(
                (norm_var_image[:, :, 0] == 1.0) &
                (norm_var_image[:, :, 1] == 1.0) &
                (norm_var_image[:, :, 2] == 1.0)
            )
        ]

        norm_var_single_value = \
            np.mean(norm_var_valid_values, axis=0).reshape(3, 1, 1)

        # Create statistic images
        depth_mean_image = \
            depth_mean_single_value * np.ones(image_shape, dtype=np.float32)
        depth_var_image = \
            depth_var_single_value * np.ones(image_shape, dtype=np.float32)

        norm_mean_image = \
            norm_mean_single_value * np.ones(image_shape, dtype=np.float32)
        norm_var_image = \
            norm_var_single_value * np.ones(image_shape, dtype=np.float32)

        means = np.concatenate(
            [
                rgb_mean,
                depth_mean_image,
                norm_mean_image,
                flow_mean
            ]
        )

        vars = np.concatenate(
            [
                rgb_var,
                depth_var_image,
                norm_var_image,
                flow_var
            ]
        )

        print(means.shape)
        print(vars.shape)
        print(means)
        print(vars)

        h5_file.create_dataset(
            name="mean",
            shape=means.shape,
            dtype=means.dtype,
            data=means
        )

        h5_file.create_dataset(
            name="var",
            shape=vars.shape,
            dtype=vars.dtype,
            data=vars
        )


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
        add_mean_and_var(input_loc)
    except ValueError as e:
        print("Error encountered: {}".format(e))
        sys.exit(1)
