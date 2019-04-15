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


def fix_strided_dataset(original_h5_loc: os.path,
                        strided_h5_loc: os.path,
                        start_x_crop: int,
                        stop_x_crop: int,
                        start_y_crop: int,
                        stop_y_crop: int):
    flow_key = 'flowuint8'
    with h5py.File(original_h5_loc, 'r') as original_h5:
        input_flow_dataset = original_h5[flow_key]

        with h5py.File(strided_h5_loc, 'a') as strided_h5:
            stride = len([k for k in strided_h5.keys() if k[:7] == 'rgbflow'])

            num_samples = input_flow_dataset.shape[0]

            output_datasets = [
                strided_h5['rgbflow' + str(stride_idx)]
                for stride_idx in range(stride)
            ]

            print("Extracting {} dataset".format(flow_key))
            bar = progressbar.ProgressBar()
            for sample_idx in bar(range(num_samples)):
                flow = input_flow_dataset[
                       sample_idx,
                       :,
                       start_y_crop: stop_y_crop,
                       start_x_crop: stop_x_crop
                       ]

                dataset_idx = sample_idx % stride

                output_datasets[dataset_idx][sample_idx // stride, 3:] = flow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add mid and custom granularities to a dataset, "
                    "based on the existing fine granularity."
    )

    parser.add_argument("original",
                        help="The input h5 file with flowuint8 dataset.",
                        type=str)

    parser.add_argument("strided",
                        help="The output h5 file to be fixed.",
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

    args = parser.parse_args()

    if os.path.isfile(args.input):
        fix_strided_dataset(
            args.original,
            args.strided,
            args.startx,
            args.stopx,
            args.starty,
            args.stopy
        )
    else:
        print("File does not exist at location, exiting...")
        sys.exit(1)
