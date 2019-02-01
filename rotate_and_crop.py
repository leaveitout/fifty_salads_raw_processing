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
import open3d
from pypcd.pypcd import PointCloud
import progressbar
import h5py
from skimage.transform import rotate

import welford


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


def load_color(image_loc: os.path):
    # load image, scale and return it as numpy array
    image = cv2.imread(image_loc)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_scaling = 65535 / 255.0
    image = image * image_scaling
    return image.astype(np.uint16)


def load_depth(image_loc: os.path):
    # load image, scale and return it as numpy array
    image = cv2.imread(image_loc, cv2.IMREAD_ANYDEPTH)
    return np.reshape(image, image.shape + (1,))


def load_flow(cloud_loc: os.path):
    if os.path.isfile(cloud_loc):
        point_cloud = open3d.read_point_cloud(cloud_loc)
        points = np.asarray(point_cloud.points)
        points = np.reshape(points, (240, 320, 3)).astype(np.float32)
        points = np.nan_to_num(points, copy=False)

        uint16_upper = 65535
        min_flow = -0.1
        max_flow = 0.1
        flow_diff = max_flow - min_flow

        cloud_repr = points - min_flow
        cloud_repr = np.clip(cloud_repr, 0.0, flow_diff)
        cloud_repr *= (uint16_upper / flow_diff)

        return cloud_repr.astype(np.uint16)
    else:
        return np.zeros((240, 320, 3), dtype=np.float16)


def load_norms_and_curvature(cloud_loc: os.path):
    cloud = PointCloud.from_path(cloud_loc)

    norms = np.stack((np.nan_to_num(cloud.pc_data['normal_x'], copy=False),
                      np.nan_to_num(cloud.pc_data['normal_y'], copy=False),
                      np.nan_to_num(cloud.pc_data['normal_z'], copy=False)),
                     axis=-1)

    norms += 1.0
    norm_scaling = 65535 / 2.0
    norms *= norm_scaling
    norms = norms.astype(np.uint16)

    curvature = np.nan_to_num(cloud.pc_data['curvature'])

    max_curvature = 0.35
    curvature_scaling = 65535 / max_curvature
    curvature *= curvature_scaling
    curvature = curvature.astype(np.uint16)

    curvature = np.reshape(curvature, curvature.shape + (1,))

    norms_curvature = np.concatenate((norms, curvature), axis=-1)

    return norms_curvature


def rotate_and_crop(image: np.ndarray,
                    angle: float,
                    start_x_crop: int,
                    stop_x_crop: int,
                    start_y_crop: int,
                    stop_y_crop: int) -> np.ndarray:
    # rotated_image = image
    # rotated_image = rotate(image, angle, preserve_range=True).astype(np.uint16)
    return image[start_y_crop: stop_y_crop, start_x_crop: stop_x_crop]


def get_splits(offsets, num_splits) -> np.ndarray:
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


def from_nhwc_to_nchw(input_data):
    """
    Converts a numpy array from xyzhwc to xyzchw.

    :param input_data: Input array with channels last.
    :return: Output array with channels third from the end.
    """
    return np.moveaxis(input_data, -1, -3)


def check_files_correct(rgb_files,
                        depth_files,
                        flow_files,
                        norm_files,
                        sample_dir):
    # Sense check the files
    for rgb_file, depth_file, flow_file, norm_file in \
            zip(rgb_files, depth_files, flow_files, norm_files):
        if rgb_file.split('.')[0] != depth_file.split('.')[0]:
            raise ValueError("{} does not match {} "
                             "for sample dir {}".format(rgb_file,
                                                        depth_file,
                                                        sample_dir))

        if rgb_file.split('.')[0] != flow_file.split('.')[0]:
            raise ValueError("{} does not match {} "
                             "for sample dir {}".format(rgb_file,
                                                        flow_file,
                                                        sample_dir))

        if norm_file.split('.')[0] != norm_file.split('.')[0]:
            raise ValueError("{} does not match {} "
                             "for sample dir {}".format(rgb_file,
                                                        norm_file,
                                                        sample_dir))

    if len(rgb_files) != len(depth_files):
        raise ValueError("Sample dir {} does not contain equal "
                         "number of rgb and depth files".format(sample_dir))

    if len(rgb_files) != len(flow_files) + 1:
        raise ValueError("Sample dir {} does not contain equal "
                         "number of rgb and depth files".format(sample_dir))

    if len(rgb_files) != len(norm_files):
        raise ValueError("Sample dir {} does not contain equal "
                         "number of rgb and norm files".format(sample_dir))


def add_mean_and_var(input_file_location: os.path) -> None:
    means = []
    vars = []

    with h5py.File(input_file_location, 'r+') as h5_file:
        splits = h5_file['splits'][:]
        offsets = h5_file['offsets'][:]
        frames = h5_file['frame']
        num_splits = len(splits)

        for split_idx in range(num_splits):
            print("Split {} of {}:".format(split_idx + 1, num_splits))
            sample_include = splits[split_idx]
            train_idxs = []

            for idx, inc in enumerate(sample_include):
                sample_idxs = list(range(offsets[idx], offsets[idx + 1]))

                if inc:
                    train_idxs += sample_idxs

            frame_accum = welford.WelfordAccumulator(min_valid_samples=100)

            bar = progressbar.ProgressBar()
            for train_idx in bar(train_idxs):
                frame = frames[train_idx]

                # The entire dataset is covered after two splits.
                if split_idx < 2:
                    frame = np.nan_to_num(frame, copy=False)
                    frames[train_idx] = frame
                    # flow_image = np.nan_to_num(flow_image)
                    # flow[train_idx] = flow_image

                frame_accum.add(frame)

            split_mean = frame_accum.get_mean().astype(np.float32)
            split_var = frame_accum.get_variance().astype(np.float32)

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


def process_sample(sample_dir: os.path,
                   frame_dataset: h5py.Dataset,
                   fine_dataset: h5py.Dataset,
                   mid_dataset: h5py.Dataset,
                   coarse_dataset: h5py.Dataset,
                   custom_dataset: h5py.Dataset,
                   offset: int,
                   start_x_crop: int,
                   stop_x_crop: int,
                   start_y_crop: int,
                   stop_y_crop: int,
                   rotate_angle: int,
                   downscale: float):
    rgb_dir = os.path.join(sample_dir, 'rgb')
    depth_dir = os.path.join(sample_dir, 'depth_f')
    flow_dir = os.path.join(sample_dir, 'flow')
    norm_dir = os.path.join(sample_dir, 'pcd')

    rgb_files = glob.glob1(rgb_dir, '*.png')
    rgb_files.sort()
    depth_files = glob.glob1(depth_dir, '*.png')
    depth_files.sort()
    flow_files = glob.glob1(flow_dir, '*.pcd')
    flow_files.sort()
    norm_files = glob.glob1(norm_dir, '*.pcd')
    norm_files.sort()

    check_files_correct(rgb_files,
                        depth_files,
                        flow_files,
                        norm_files,
                        sample_dir)

    print("Processing frames...")
    bar = progressbar.ProgressBar()

    for idx, image_tuple in \
            bar(enumerate(zip(rgb_files, depth_files, flow_files, norm_files))):
        rgb_file, depth_file, flow_file, norm_file = image_tuple
        rgb_file_path = os.path.join(rgb_dir, rgb_file)
        rgb_mat = load_color(rgb_file_path)

        depth_file_path = os.path.join(depth_dir, depth_file)
        depth_mat = load_depth(depth_file_path)
        depth_mat = np.nan_to_num(depth_mat, copy=False)

        flow_file_path = os.path.join(flow_dir, flow_file)
        flow_mat = load_flow(flow_file_path)

        norm_file_path = os.path.join(norm_dir, norm_file)
        norm_mat = load_norms_and_curvature(norm_file_path)

        frame = np.concatenate((rgb_mat, depth_mat, flow_mat, norm_mat),
                               axis=-1)

        # Rotate and crop
        frame = rotate_and_crop(frame, rotate_angle, start_x_crop, stop_x_crop,
                                start_y_crop, stop_y_crop)

        # Downscale
        if 0.0 < downscale < 1.0:
            step = int(1 / downscale)
            frame = frame[::step, ::step, :]

        # Permute dims
        frame = from_nhwc_to_nchw(frame)

        # Add to dataset.
        frame_dataset[offset + idx] = frame
        # np.save('/media/rapid/test.npy', frame)

    fine_labels_file = os.path.join(sample_dir, 'labels_fine.npy')
    fine_labels = np.load(fine_labels_file)
    fine_dataset[offset: offset + fine_labels.shape[0]] = fine_labels

    mid_labels_file = os.path.join(sample_dir, 'labels_mid.npy')
    mid_labels = np.load(mid_labels_file)
    mid_dataset[offset: offset + mid_labels.shape[0]] = mid_labels

    coarse_labels_file = os.path.join(sample_dir, 'labels_coarse.npy')
    coarse_labels = np.load(coarse_labels_file)
    coarse_dataset[offset: offset + coarse_labels.shape[0]] = coarse_labels

    custom_labels_file = os.path.join(sample_dir, 'labels_custom.npy')
    custom_labels = np.load(custom_labels_file)
    custom_dataset[offset: offset + custom_labels.shape[0]] = custom_labels


def process_dataset(h5_dataset_path: os.path,
                    dataset_path: os.path,
                    downscale: float = 1.0,
                    start_x_crop: int = 0,
                    stop_x_crop: int = 320,
                    start_y_crop: int = 0,
                    stop_y_crop: int = 0,
                    rotate_angle: int = 0):
    # Get the sizes for the datasets.
    sample_dirs = glob.glob1(dataset_path, "[0-9][0-9]-[0-9]")
    if len(sample_dirs) != 50:
        raise ValueError("There should be 50 sample folders, "
                         "found {}.".format(len(sample_dirs)))

    sample_dirs.sort()
    sample_dirs = [os.path.join(dataset_path, s) for s in sample_dirs]
    sample_sizes = [np.load(os.path.join(sd, 'labels_coarse.npy')).shape[0]
                    for sd in sample_dirs]
    sample_sizes.insert(0, 0)

    # Get the offsets of the datasets.
    sample_offsets = np.cumsum(sample_sizes)

    # Get the splits of the datasets.
    sample_splits = get_splits(sample_offsets, 5)

    h5_file = h5py.File(h5_dataset_path, 'w')

    def calc_dimension(start, stop, scale):
        return int(scale * (stop - start))

    height = calc_dimension(start_y_crop, stop_y_crop, downscale)
    width = calc_dimension(start_x_crop, stop_x_crop, downscale)

    # R-G-B-D-Fx-Fy-Fz-Nx-Ny-Nz-C
    num_image_types = 11
    total_samples = sample_offsets[-1]
    frame_shape = (num_image_types, height, width)
    frame_dataset_shape = (total_samples,) + frame_shape
    frame_chunk_shape = (1,) + frame_shape

    num_fine_actions = 52
    num_mid_actions = 18
    num_coarse_actions = 4
    num_custom_actions = 10
    fine_dataset_shape = (sample_offsets[-1], num_fine_actions)
    mid_dataset_shape = (sample_offsets[-1], num_mid_actions)
    coarse_dataset_shape = (sample_offsets[-1], num_coarse_actions)
    custom_dataset_shape = (sample_offsets[-1], num_custom_actions)

    # Set up the datasets within the h5file.
    frame_dataset = h5_file.create_dataset(name='frame',
                                           shape=frame_dataset_shape,
                                           dtype=np.uint16,
                                           compression='lzf',
                                           chunks=frame_chunk_shape)

    fine_dataset = h5_file.create_dataset(name='fine',
                                          shape=fine_dataset_shape,
                                          dtype=np.float32)

    mid_dataset = h5_file.create_dataset(name='mid',
                                         shape=mid_dataset_shape,
                                         dtype=np.float32)

    coarse_dataset = h5_file.create_dataset(name='coarse',
                                            shape=coarse_dataset_shape,
                                            dtype=np.float32)

    custom_dataset = h5_file.create_dataset(name='custom',
                                            shape=custom_dataset_shape,
                                            dtype=np.float32)

    _ = h5_file.create_dataset(name='splits',
                               shape=sample_splits.shape,
                               dtype=sample_splits.dtype,
                               data=sample_splits)

    _ = h5_file.create_dataset(name='offsets',
                               shape=sample_offsets.shape,
                               dtype=sample_offsets.dtype,
                               data=sample_offsets)

    for sample_dir, offset in zip(sample_dirs, sample_offsets):
        print("Processing sample dir: {}".format(sample_dir))
        process_sample(sample_dir,
                       frame_dataset,
                       fine_dataset,
                       mid_dataset,
                       coarse_dataset,
                       custom_dataset,
                       offset,
                       start_x_crop,
                       stop_x_crop,
                       start_y_crop,
                       stop_y_crop,
                       rotate_angle,
                       downscale)

    h5_file.close()

    add_mean_and_var(h5_dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a h5 dataset from a 50Salads preprocessed folder, "
                    "rotating and cropping the images.")
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
    parser.add_argument("--start_x",
                        help="Start of x crop.",
                        default=0,
                        type=int)
    parser.add_argument("--stop_x",
                        help="Stop of x crop.",
                        default=320,
                        type=int)
    parser.add_argument("--start_y",
                        help="Start of y crop.",
                        default=0,
                        type=int)
    parser.add_argument("--stop_y",
                        help="Stop of y crop.",
                        default=240,
                        type=int)
    parser.add_argument("--rotate",
                        help="Angle to rotate by.",
                        default=1,
                        type=int)
    args = parser.parse_args()

    try:
        dataset_path = os.path.abspath(args.input)
        output_h5_path = os.path.abspath(args.output)
        output_h5_parent_path = os.path.dirname(output_h5_path)

        if not os.path.isdir(dataset_path):
            raise ValueError("Invalid directory for h5 dataset.")
        if not os.path.isdir(output_h5_parent_path):
            raise ValueError("Invalid directory for h5 dataset.")
        elif os.path.isfile(output_h5_path):
            print("Output h5 dataset file already exists, overwriting.")

        process_dataset(h5_dataset_path=output_h5_path,
                        dataset_path=dataset_path,
                        downscale=args.downscale,
                        start_x_crop=args.start_x,
                        stop_x_crop=args.stop_x,
                        start_y_crop=args.start_y,
                        stop_y_crop=args.stop_y,
                        rotate_angle=args.rotate)

    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
