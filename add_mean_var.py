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
    means = []
    vars = []

    with h5py.File(input_file_location, 'r+') as h5_file:
        splits = h5_file['splits'][:]
        offsets = h5_file['offsets'][:]
        rgb = h5_file['rgb']
        depth = h5_file['depth']
        depth_f = h5_file['depth_f']
        flow = h5_file['flow']
        num_splits = len(splits)

        for split_idx in range(num_splits):
            print("Split {} of {}:".format(split_idx + 1, num_splits))
            sample_include = splits[split_idx]
            train_idxs = []

            for idx, inc in enumerate(sample_include):
                sample_idxs = list(range(offsets[idx], offsets[idx + 1]))

                if inc:
                    train_idxs += sample_idxs

            rgb_accum = welford.WelfordAccumulator(min_valid_samples=100)
            depth_accum = welford.WelfordAccumulator(min_valid_samples=100)
            depth_f_accum = welford.WelfordAccumulator(min_valid_samples=100)
            flow_accum = welford.WelfordAccumulator(min_valid_samples=100)

            bar = progressbar.ProgressBar()
            for train_idx in bar(train_idxs):
                rgb_image = rgb[train_idx]
                depth_image = depth[train_idx]
                depth_f_image = depth_f[train_idx]
                flow_image = flow[train_idx]

                # The entire dataset is covered after two splits.
                if split_idx < 2:
                    depth_image = np.nan_to_num(depth_image, copy=False)
                    depth[train_idx] = depth_image
                    depth_f_image = np.nan_to_num(depth_f_image, copy=False)
                    depth_f[train_idx] = depth_f_image
                    # flow_image = np.nan_to_num(flow_image)
                    # flow[train_idx] = flow_image

                rgb_accum.add(rgb_image)
                depth_accum.add(depth_image)
                depth_f_accum.add(depth_f_image)
                flow_accum.add(flow_image)

            rgb_mean = rgb_accum.get_mean().astype(np.float32)
            rgb_var = rgb_accum.get_variance().astype(np.float32)
            depth_mean = depth_accum.get_mean().astype(np.float32)
            depth_var = depth_accum.get_variance().astype(np.float32)
            depth_f_mean = depth_f_accum.get_mean().astype(np.float32)
            depth_f_var = depth_f_accum.get_variance().astype(np.float32)
            flow_mean = flow_accum.get_mean().astype(np.float32)
            flow_var = flow_accum.get_variance().astype(np.float32)

            split_mean = np.concatenate((rgb_mean,
                                         depth_mean,
                                         depth_f_mean,
                                         flow_mean))

            split_var = np.concatenate((rgb_var,
                                        depth_var,
                                        depth_f_var,
                                        flow_var))

            means.append(split_mean)
            vars.append(split_var)

        means = np.array(means)
        vars = np.array(vars)
        print(means.shape)
        print(vars.shape)
        print(means[0])
        print(vars[0])

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
        description="Calculate and add the mean and var for each split")
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
