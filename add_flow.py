#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import numpy as np
import h5py
import progressbar


def add_flow_uint8_to_dataset(h5_dataset_path: os.path):
    with h5py.File(h5_dataset_path, 'r+') as h5_file:
        flow_dataset = h5_file['flow']
        flow_chunk_shape = (1,) + flow_dataset.shape[1:]

        flow_uint8_dataset = h5_file.create_dataset(name='flowuint8',
                                                    shape=flow_dataset.shape,
                                                    dtype=np.uint8,
                                                    compression='lzf',
                                                    chunks=flow_chunk_shape)

        max_abs_flow = 0.05
        distinct_values = 256

        flow_increments = max_abs_flow / (distinct_values // 2)
        max_flow = max_abs_flow
        min_flow = max_flow - (flow_increments * (distinct_values - 1))
        scale = (distinct_values - 1) / (max_flow - min_flow)

        bar = progressbar.ProgressBar()
        for idx in bar(np.arange(flow_dataset.shape[0])):
            flow_uint8_dataset[idx] = \
                convert_flow_image(flow_dataset[idx], min_flow, max_flow, scale)


def convert_flow_image(input_flow_image: np.ndarray, min_flow, max_flow, scale):
    rescaled_flow_image = np.clip(input_flow_image, min_flow, max_flow)
    rescaled_flow_image += min_flow
    rescaled_flow_image *= scale
    return rescaled_flow_image.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a lower precision flow h5 dataset.")
    parser.add_argument("input", help="The input h5 file.", type=str)
    args = parser.parse_args()

    try:
        dataset_path = os.path.abspath(args.input)

        if not os.path.isfile(dataset_path):
            raise ValueError("Invalid h5 dataset location.")

        add_flow_uint8_to_dataset(dataset_path)
    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
