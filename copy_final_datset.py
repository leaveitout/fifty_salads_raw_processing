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


def copy_chunked_dataset(input_h5_path: os.path,
                         output_h5_path: os.path,
                         input_dataset_key: str,
                         output_dataset_key: str):
    with h5py.File(input_h5_path, 'r') as input_h5_file:
        with h5py.File(output_h5_path, 'a') as output_h5_file:
            input_dataset = input_h5_file[input_dataset_key]

            output_dataset = output_h5_file.create_dataset(
                output_dataset_key,
                shape=input_dataset.shape,
                dtype=input_dataset.dtype,
                chunks=(1,) + input_dataset.shape[-3:],
                compression='lzf'
            )

            print("Copying {} dataset".format(output_dataset_key))
            bar = progressbar.ProgressBar()
            for idx in bar(range(input_dataset.shape[0])):
                output_dataset[idx] = input_dataset[idx]


def copy_dataset(input_h5_path: os.path,
                 output_h5_path: os.path,
                 input_dataset_key: str,
                 output_dataset_key: str):
    with h5py.File(input_h5_path, 'r') as input_h5_file:
        with h5py.File(output_h5_path, 'a') as output_h5_file:
            print("Copying {} dataset".format(output_dataset_key))
            output_h5_file.create_dataset(
                output_dataset_key,
                data=input_h5_file[input_dataset_key][:]
            )


def copy_final_datasets(input_h5_path: os.path, output_h5_path: os.path):
    copy_chunked_dataset(input_h5_path, output_h5_path, 'rgb', 'rgb')
    copy_chunked_dataset(input_h5_path, output_h5_path, 'depth_jet', 'depth')
    copy_chunked_dataset(input_h5_path, output_h5_path, 'norm', 'norm')
    copy_chunked_dataset(input_h5_path, output_h5_path, 'flow', 'flow')

    copy_dataset(input_h5_path, output_h5_path, 'coarse', 'coarse')
    copy_dataset(input_h5_path, output_h5_path, 'custom', 'custom')
    copy_dataset(input_h5_path, output_h5_path, 'mid', 'mid')
    copy_dataset(input_h5_path, output_h5_path, 'fine', 'fine')
    copy_dataset(input_h5_path, output_h5_path, 'splits', 'splits')
    copy_dataset(input_h5_path, output_h5_path, 'offsets', 'offsets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Final config dataset output to a new clean dataset.")
    parser.add_argument(
        "input",
        help="The input h5 file. Should contain a dataset called 'depth_jet' "
             "of type uint8. Depth and norm datasets should have all invalid "
             "values added with mean. The mean and variance should be constant "
             "across the mean and variance images.",
        type=str
    )
    parser.add_argument(
        "output",
        help="The output h5 file.",
        type=str
    )
    args = parser.parse_args()

    try:
        input_h5_file_path = os.path.abspath(args.input)

        if not os.path.isfile(input_h5_file_path):
            raise ValueError("Invalid h5 dataset location.")

        output_h5_file_path = os.path.abspath(args.output)
        output_h5_parent_path = os.path.dirname(output_h5_file_path)

        if not os.path.isdir(output_h5_parent_path):
            raise ValueError("Invalid directory for h5 dataset.")
        elif os.path.isfile(output_h5_file_path):
            print("Output h5 dataset file already exists, overwriting.")

        copy_final_datasets(
            input_h5_file_path,
            output_h5_file_path
        )
    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
