#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2018.
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import glob
import numpy as np
import cv2
import pcl
import h5py
import progressbar


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
    # image = cv2.resize(image, )
    # TODO: To remove, bug is in process.py that leaves last file full size
    # if image.shape[0] != 240:
    #     image = image[::2, ::2, ::]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_cloud(cloud_mat: np.ndarray,
                 downscale: float,
                 cv2_interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    # if cv2_interpolation == cv2.INTER_NEAREST:
    #     return cloud_mat[::2, ::2, :].astype(np.float16)
    # else:
    cloud_planes = np.split(cloud_mat, 3, axis=-1)

    cloud_planes_resized = []

    for cp in cloud_planes:
        cp_no_nan = np.nan_to_num(cp, copy=False)
        cp_resized = cv2.resize(
            cp_no_nan,
            dsize=None,
            fx=downscale,
            fy=downscale,
            interpolation=cv2_interpolation
        ).astype(np.float16)

        cp_resized.resize((cp_resized.shape[0], cp_resized.shape[1], 1))

        cloud_planes_resized.append(cp_resized)

    return np.concatenate(cloud_planes_resized, axis=-1)


def resize_image_float16(image_mat: np.ndarray,
                         downscale: float,
                         cv2_interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    image_mat_typed = image_mat.astype(np.float32)

    image_mat_downscaled = cv2.resize(image_mat_typed,
                                      dsize=None,
                                      fx=downscale,
                                      fy=downscale,
                                      interpolation=cv2_interpolation)

    return image_mat_downscaled.astype(np.float16)


def resize_image(image_mat: np.ndarray,
                 downscale: float,
                 cv2_interpolation=cv2.INTER_AREA) -> np.ndarray:
    resized = cv2.resize(image_mat,
                         dsize=None,
                         fx=downscale,
                         fy=downscale,
                         interpolation=cv2_interpolation)

    if len(resized.shape) == 2:
        resized.resize((resized.shape[0], resized.shape[1], 1))

    return resized


def load_depth_pgm(depth_pgm_loc):
    # TODO: Need to downsize as it will be too large
    with open(depth_pgm_loc, 'rb') as pgm_file:
        pgm_dims = bytearray(pgm_file.readline())
        pgm_dims = pgm_dims.decode('utf-8').split()
        print(pgm_dims)
        height = int(pgm_dims[1])
        width = int(pgm_dims[2])
        max_val = int(pgm_dims[3])
        print(height, width, max_val)
        depth = np.fromfile(pgm_file, dtype=np.uint16, count=height * width)
        depth = depth.reshape((width, height))
        return depth


def load_depth_image(depth_loc):
    """
    Return the depth image at the location as as float32 numpy array.

    :param depth_loc: The location of the depth image.
    :return: A np.float32 numpy array of the depth image.
    """
    im = cv2.imread(depth_loc)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return np.reshape(im, (240, 320, 1)).astype(np.uint8)


def load_cloud(cloud_loc):
    if os.path.isfile(cloud_loc):
        return np.reshape(pcl.load(cloud_loc).to_array(), (240, 320, 3))
    else:
        return np.zeros((240, 320, 3), dtype=np.float32)


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
        splits[row, test_sample_start: test_sample_start + samples_per_split] \
            = False

    return splits


def from_nhwc_to_nchw(input_data: np.ndarray):
    """
    Converts a numpy array from xyzhwc to xyzchw.

    :param input_data: Input array with channels last.
    :return: Output array with channels third from the end.
    """
    return np.rollaxis(input_data, -1, -3)


def check_files_correct(rgb_files, depth_pgm_files, flow_files, sample_dir):
    # Sense check the files
    for rgb_file, depth_pgm_file, flow_file in \
            zip(rgb_files, depth_pgm_files, flow_files):
        if rgb_file.split('.')[0] != depth_pgm_file.split('.')[0]:
            raise ValueError("{} does not match {} "
                             "for sample dir {}".format(rgb_file,
                                                        depth_pgm_file,
                                                        sample_dir))

        if rgb_file.split('.')[0] != flow_file.split('.')[0]:
            raise ValueError("{} does not match {} "
                             "for sample dir {}".format(rgb_file,
                                                        flow_file,
                                                        sample_dir))

    if len(rgb_files) != len(depth_pgm_files):
        raise ValueError("Sample dir {} does not contain equal "
                         "number of rgb and depth files".format(sample_dir))

    if len(rgb_files) != len(flow_files) + 1:
        raise ValueError("Sample dir {} does not contain equal "
                         "number of rgb and depth files".format(sample_dir))


def process_sample(sample_dir: os.path,
                   rgbd_dataset: h5py.Dataset,
                   flow_dataset: h5py.Dataset,
                   fine_dataset: h5py.Dataset,
                   mid_dataset: h5py.Dataset,
                   coarse_dataset: h5py.Dataset,
                   custom_dataset: h5py.Dataset,
                   offset: int,
                   downscale: float):
    print("Processing sample: {}".format(sample_dir))

    rgb_dir = os.path.join(sample_dir, 'rgb')
    depth_dir = os.path.join(sample_dir, 'depth')
    flow_dir = os.path.join(sample_dir, 'flow')

    rgb_files = glob.glob1(rgb_dir, '*.png')
    rgb_files.sort()
    depth_files = glob.glob1(depth_dir, '*.png')
    depth_files.sort()
    flow_files = glob.glob1(flow_dir, '*.pcd')
    flow_files.sort()

    check_files_correct(rgb_files, depth_files, flow_files, sample_dir)

    print("Processing rgbd...")

    bar = progressbar.ProgressBar(max_value=len(rgb_files))
    for idx, rgbd_tuple in bar(enumerate(zip(rgb_files, depth_files))):
        rgb_file_path, depth_file_path = rgbd_tuple
        rgb_file_path = os.path.join(rgb_dir, rgb_file_path)
        rgb_mat = load_image(rgb_file_path)

        if 0.0 < downscale < 1.0:
            rgb_mat = resize_image(rgb_mat, downscale)

        rgb_mat_chw = from_nhwc_to_nchw(rgb_mat)

        depth_file_path = os.path.join(depth_dir, depth_file_path)
        depth_mat = load_depth_image(depth_file_path)

        if 0.0 < downscale < 1.0:
            depth_mat = resize_image(depth_mat, downscale, cv2.INTER_NEAREST)

        depth_mat_chw = from_nhwc_to_nchw(depth_mat)

        combined_mat = np.concatenate((rgb_mat_chw, depth_mat_chw), axis=0)

        rgbd_dataset[offset + idx] = combined_mat

    print("Processing flow...")

    bar = progressbar.ProgressBar(max_value=len(flow_files))
    for idx, flow_file in bar(enumerate(flow_files)):
        flow_file_path = os.path.join(flow_dir, flow_file)
        flow_mat = load_cloud(flow_file_path)

        if 0.0 < downscale < 1.0:
            flow_mat = resize_cloud(flow_mat, downscale)

        flow_mat_chw = from_nhwc_to_nchw(flow_mat)

        flow_dataset[offset + idx] = flow_mat_chw

    print("Processing labels")

    # TODO: Put this in a function
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

    print("Done!")


def process_dataset(h5_dataset_path: os.path,
                    dataset_path: os.path,
                    downscale: float = 1.0):
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

    frame_shape = (int(240 * downscale), int(320 * downscale))
    rgbd_dataset_shape = (sample_offsets[-1], 4, frame_shape[0], frame_shape[1])
    flow_dataset_shape = (sample_offsets[-1], 3, frame_shape[0], frame_shape[1])
    rgbd_chunk_shape = (1,) + rgbd_dataset_shape[-3:]
    flow_chunk_shape = (1,) + flow_dataset_shape[-3:]

    num_fine_actions = 52
    num_mid_actions = 18
    num_coarse_actions = 4
    num_custom_actions = 10
    fine_dataset_shape = (sample_offsets[-1], num_fine_actions)
    mid_dataset_shape = (sample_offsets[-1], num_mid_actions)
    coarse_dataset_shape = (sample_offsets[-1], num_coarse_actions)
    custom_dataset_shape = (sample_offsets[-1], num_custom_actions)

    # Set up the datasets within the h5file.
    rgbd_dataset = h5_file.create_dataset(name='rgbd',
                                          shape=rgbd_dataset_shape,
                                          dtype=np.uint8,
                                          compression='lzf',
                                          chunks=rgbd_chunk_shape)

    flow_dataset = h5_file.create_dataset(name='flow',
                                          shape=flow_dataset_shape,
                                          dtype=np.float16,
                                          compression='lzf',
                                          chunks=flow_chunk_shape)

    fine_dataset = h5_file.create_dataset(name='fine',
                                          shape=fine_dataset_shape,
                                          dtype=np.float)

    mid_dataset = h5_file.create_dataset(name='mid',
                                         shape=mid_dataset_shape,
                                         dtype=np.float)

    coarse_dataset = h5_file.create_dataset(name='coarse',
                                            shape=coarse_dataset_shape,
                                            dtype=np.float)

    custom_dataset = h5_file.create_dataset(name='custom',
                                            shape=custom_dataset_shape,
                                            dtype=np.float)

    _ = h5_file.create_dataset(name='splits',
                               shape=sample_splits.shape,
                               dtype=sample_splits.dtype,
                               data=sample_splits)

    _ = h5_file.create_dataset(name='offsets',
                               shape=sample_offsets.shape,
                               dtype=sample_offsets.dtype,
                               data=sample_offsets)

    for sample_dir, offset in zip(sample_dirs, sample_offsets):
        process_sample(sample_dir,
                       rgbd_dataset,
                       flow_dataset,
                       fine_dataset,
                       mid_dataset,
                       coarse_dataset,
                       custom_dataset,
                       offset,
                       downscale)

    # TODO: Calculate the mean and var for each split using the sample_splits
    # rgb = h5_file.create_dataset('rgb', shape=sample_offsets[-1])

    # Process the frames
    # print("Processing the frame data...")
    # bar = progressbar.ProgressBar()
    # counts = []
    # for sd in bar(sample_dirs):
    #     counts.append(process_sample_folder(h5_file, sd, downscale))
    #
    # count_sum = accumulate(counts)
    #
    # # Process the labels
    # offsets_gran = []
    # for gran in ['coarse', 'mid', 'fine']:
    #     all_labels = None
    #     offsets = [0]
    #     for sd in sample_dirs:
    #         labels = get_labels(sd, gran)
    #         if all_labels is None:
    #             all_labels = labels
    #         else:
    #             all_labels = np.concatenate((all_labels, labels))
    #         offsets.append(all_labels.shape[0])
    #     offsets_gran.append(offsets)
    #
    #     h5_file.create_dataset(gran, data=all_labels)
    #
    # compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    #
    # for offsets in offsets_gran:
    #     if not compare(count_sum, offsets):
    #         print("The number of sample frames and labels do not match.")
    #         print(counts)
    #         print(offsets)
    #         break
    # else:
    #     h5_file.create_dataset('offsets', data=offsets_gran[0])
    #
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a h5 dataset from a 50Salads preprocessed folder.")
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
    args = parser.parse_args()

    try:
        input_dataset_path = os.path.abspath(args.input)
        output_h5_path = os.path.abspath(args.output)
        output_h5_parent_path = os.path.dirname(output_h5_path)

        if not os.path.isdir(input_dataset_path):
            raise ValueError("Invalid directory for h5 dataset.")
        if not os.path.isdir(output_h5_parent_path):
            raise ValueError("Invalid directory for h5 dataset.")
        elif os.path.isfile(output_h5_path):
            print("Output h5 dataset file already exists, overwriting.")

        process_dataset(h5_dataset_path=output_h5_path,
                        dataset_path=input_dataset_path,
                        downscale=args.downscale)

    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("OS Error: {}".format(err))
        sys.exit(1)
