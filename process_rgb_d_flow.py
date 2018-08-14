#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import sys
import os
import glob
import subprocess
import progressbar
import multiprocessing as mp
import functools


def extract_depth_images(pcd_dir: os.path,
                         output_depth_path: os.path):
    os.mkdir(output_depth_path)

    pcd_filenames = [pcd_file for pcd_file in os.listdir(pcd_dir)
                     if os.path.isfile(os.path.join(pcd_dir, pcd_file))]

    bar = progressbar.ProgressBar()
    for pcd_filename in bar(pcd_filenames):
        depth_filename = pcd_filename.split('.')[0] + '.png'
        depth_file_path = os.path.join(output_depth_path, depth_filename)
        pcd_file_path = os.path.join(pcd_dir, pcd_filename)
        extract_depth_cmd = ['pcl_pcd2png',
                             '--no-nan',
                             '--field',
                             'z',
                             pcd_file_path,
                             depth_file_path]

        subprocess.call(extract_depth_cmd,
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL)


def calc_scene_flow(pcd_dir: os.path,
                    output_scene_flow_path: os.path):
    if not os.path.exists(output_scene_flow_path) \
            and not os.path.isdir(output_scene_flow_path):
        os.mkdir(output_scene_flow_path)

    scene_flow_cmd = ['pcl_pd_flow',
                      '-p',
                      str(pcd_dir),
                      '-d',
                      str(output_scene_flow_path)]
    subprocess.call(scene_flow_cmd,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL)


def hole_filling_sample_dir(sample_dir: os.path):
    pcd_dir = os.path.join(sample_dir, 'pcd')
    filtered_pcd_dir = os.path.join(sample_dir, 'pcd_f')
    filtered_depth_dir = os.path.join(sample_dir, 'depth_f')
    flow_dir = os.path.join(sample_dir, 'flow')

    if not os.path.exists(filtered_pcd_dir) \
            and not os.path.isdir(filtered_pcd_dir):
        filter_pcd_cmd = ['pcl_video_hole_filter',
                          str(pcd_dir),
                          str(filtered_pcd_dir)]
        print("Filtering sample {} into {}".format(sample_dir,
                                                   filtered_pcd_dir))
        subprocess.call(filter_pcd_cmd)

    if not os.path.exists(filtered_depth_dir) \
            and not os.path.isdir(filtered_depth_dir):
        print("Extracting depth into {}".format(filtered_depth_dir))
        extract_depth_images(filtered_pcd_dir, filtered_depth_dir)

    print("Calculating scene flow {}".format(flow_dir))
    calc_scene_flow(filtered_pcd_dir, flow_dir)


def hole_filling_sample_dir_threadsafe(sample_dir: os.path):
    pcd_dir = os.path.join(sample_dir, 'pcd')
    filtered_pcd_dir = os.path.join(sample_dir, 'pcd_f')
    filtered_depth_dir = os.path.join(sample_dir, 'depth_f')
    flow_dir = os.path.join(sample_dir, 'flow')

    if not os.path.exists(filtered_pcd_dir) \
            and not os.path.isdir(filtered_pcd_dir):
        filter_pcd_cmd = ['pcl_video_hole_filter',
                          str(pcd_dir),
                          str(filtered_pcd_dir)]
        lock.acquire()
        try:
            print("Filtering sample into {}".format(sample_dir))
        finally:
            lock.release()

        subprocess.call(filter_pcd_cmd)

    if not os.path.exists(filtered_depth_dir) \
            and not os.path.isdir(filtered_depth_dir):
        lock.acquire()
        try:
            print("Extracting depth into {}".format(filtered_depth_dir))
        finally:
            lock.release()
        extract_depth_images(filtered_pcd_dir, filtered_depth_dir)

    lock.acquire()
    try:
        print("Calculating scene flow {}".format(flow_dir))
    finally:
        lock.release()
    calc_scene_flow(filtered_pcd_dir, flow_dir)


def hole_filling_flow_estimation(dataset_dir: os.path):
    sample_dirs = glob.glob1(dataset_dir, '[0-9][0-9]-[1-2]')
    sample_dirs.sort()
    sample_dirs = [os.path.join(dataset_dir, sd) for sd in sample_dirs]

    # hole_fill_func = functools.partial(hole_filling_sample_dir_threadsafe,
    #                                    lock=lock)

    def init(l):
        global lock
        lock = l

    l = mp.Lock()
    pool = mp.Pool(processes=2, initializer=init, initargs=(l,))
    pool.map(hole_filling_sample_dir_threadsafe, sample_dirs)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process the 50 Salads dataset to have better flow data"
    )
    parser.add_argument('dataset_dir', help='Path to the dataset directory.')

    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)

    if not os.path.isdir(dataset_dir):
        print("The specified input dir {} does not exist.".format(dataset_dir))
        sys.exit(1)

    sys.exit(hole_filling_flow_estimation(dataset_dir))
