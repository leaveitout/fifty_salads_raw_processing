#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2018.
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import glob
import collections
from itertools import accumulate
import json
import numpy as np
import cv2
import pcl
import progressbar
import h5py


def load_or_create_dataset(dataset_loc):
    """
    Loads or creates the dataset.

    :param dataset_loc:
    :return: tuple of a dataset file pointer and a bool of whether it was
    newly created.
    """
    if os.path.isfile(dataset_loc):
        return h5py.File(dataset_loc, "r+"), False
    else:
        return h5py.File(dataset_loc, "w"), True


def load_image(image_loc):
    # load image, scale and return it as numpy array
    image = cv2.imread(image_loc)
    image = cv2.resize(image, )
    # TODO: To remove, bug is in process.py that leaves last file full size
    if image.shape[0] != 240:
        image = image[::2, ::2, ::]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_depth(depth_loc):
    """
    Return the depth image at the location as as float32 numpy array.

    :param depth_loc: The location of the depth image.
    :return: A np.float32 numpy array of the depth image.
    """
    im = cv2.imread(depth_loc)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return np.reshape(im, (240, 320, 1)).astype(np.float32)


def load_cloud(cloud_loc):
    if os.path.isfile(cloud_loc):
        return np.reshape(pcl.load(cloud_loc).to_array(), (240, 320, 3))
    else:
        return np.zeros((240, 320, 3), dtype=np.float32)


def get_frame(frame_id, dir_loc, downscale=1):
    rgb_loc = os.path.join(dir_loc, 'rgb', str(frame_id) + '.png')
    rgb = load_image(rgb_loc)

    depth_loc = os.path.join(dir_loc, 'depth', str(frame_id) + '.png')
    depth = load_depth(depth_loc)

    flow_loc = os.path.join(dir_loc, 'flow', str(frame_id) + '.pcd')
    flow = load_cloud(flow_loc)

    frame = np.concatenate((rgb, depth, flow), axis=-1)

    if downscale > 1:
        frame = frame[::int(downscale), ::int(downscale), :]

    return frame


def load_labels_as_numpy(json_loc):
    with open(json_loc, 'r') as json_fp:
        return np.array(json.load(json_fp))


def process_sample_folder(dataset, dir_loc, downscale=1):
    rgb_files = glob.glob1(os.path.join(dir_loc, 'rgb'), '*.png')
    rgb_files.sort()

    # Strip the .png
    rgb_files = [r[:-4] for r in rgb_files]
    count = 0

    for f in rgb_files:
        frame = get_frame(f, dir_loc, downscale)
        # TODO: Add the frame to the dataset
        if 'frames' not in set(dataset.keys()):
            frame = np.expand_dims(frame, axis=0)
            max_shape = (None,) + frame.shape[1:]
            dataset.create_dataset('frames',
                                   data=frame,
                                   shape=frame.shape,
                                   maxshape=max_shape)
        else:
            frames_dataset = dataset['frames']
            new_frames_dataset_shape = (frames_dataset.shape[0] + 1,) + \
                                       frames_dataset.shape[1:]

            frames_dataset.resize(new_frames_dataset_shape)
            frames_dataset[-1] = frame
        count += 1

    return count


def get_labels(dir_loc, granularity: str):
    file_loc = os.path.join(dir_loc, 'labels', granularity + '_labels.json')

    return load_labels_as_numpy(file_loc)

    # TODO: Add the labels to the dataset


def process_dataset(h5_dataset, dataset_dir_loc, downscale=1):
    dataset, new_file = load_or_create_dataset(h5_dataset)

    # TODO: Set up the datasets within the h5file.
    sample_dirs = glob.glob1(dataset_dir_loc, "[0-9][0-9]-[0-9]")
    sample_dirs.sort()
    sample_dirs = [os.path.join(dataset_dir_loc, s) for s in sample_dirs]

    # Process the frames
    print("Processing the frame data...")
    bar = progressbar.ProgressBar()
    counts = []
    for sd in bar(sample_dirs):
        counts.append(process_sample_folder(dataset, sd, downscale))

    count_sum = accumulate(counts)

    # Process the labels
    offsets_gran = []
    for gran in ['coarse', 'mid', 'fine']:
        all_labels = None
        offsets = [0]
        for sd in sample_dirs:
            labels = get_labels(sd, gran)
            if all_labels is None:
                all_labels = labels
            else:
                all_labels = np.concatenate((all_labels, labels))
            offsets.append(all_labels.shape[0])
        offsets_gran.append(offsets)

        dataset.create_dataset(gran, data=all_labels)

    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    for offsets in offsets_gran:
        if not compare(count_sum, offsets):
            print("The number of sample frames and labels do not match.")
            print(counts)
            print(offsets)
            break
    else:
        dataset.create_dataset('offsets', data=offsets_gran[0])

    dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add a 50Salads preprocessed folder to h5 dataset.")
    parser.add_argument("input",
                        help="The directory containing rgb, pcd, "
                             "flow and labels folders, all processed.",
                        type=str)
    parser.add_argument("dataset_h5",
                        help="The .h5 dataset to add the sample to. "
                             "If it does not exist, it will be created.",
                        type=str)
    parser.add_argument("--downscale",
                        help="The downscale factor.",
                        default=1.0,
                        type=float)
    args = parser.parse_args()

    try:
        output_h5 = os.path.abspath(args.dataset_h5)
        output_h5_parent_path = os.path.dirname(output_h5)

        if not os.path.isdir(output_h5_parent_path):
            raise ValueError("Invalid directory for h5 dataset.")
        if os.path.isfile(output_h5):
            raise ValueError("Output h5 dataset file already exists.")

        process_dataset(h5_dataset=args.dataset_h5,
                        dataset_dir_loc=args.input,
                        downscale=args.downscale)

    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)



