#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys
import cv2
import numpy as np
import h5py
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt


# from skimage.viewer import ImageViewer
coarse_dict = {
    0: "background",
    1: "cut_and_mix_ingredients",
    2: "prepare_dressing",
    3: "serve_salad"
}

mid_dict = {
    0: "add_dressing",
    1: "add_oil",
    2: "add_pepper",
    3: "add_salt",
    4: "add_vinegar",
    5: "background",
    6: "cut_cheese",
    7: "cut_cucumber",
    8: "cut_lettuce",
    9: "cut_tomato",
    10: "mix_dressing",
    11: "mix_ingredients",
    12: "peel_cucumber",
    13: "place_cheese_into_bowl",
    14: "place_cucumber_into_bowl",
    15: "place_lettuce_into_bowl",
    16: "place_tomato_into_bowl",
    17: "serve_salad_onto_plate"
}

fine_dict = {
    0: "add_dressing_core",
    1: "add_dressing_post",
    2: "add_dressing_prep",
    3: "add_oil_core",
    4: "add_oil_post",
    5: "add_oil_prep",
    6: "add_pepper_core",
    7: "add_pepper_post",
    8: "add_pepper_prep",
    9: "add_salt_core",
    10: "add_salt_post",
    11: "add_salt_prep",
    12: "add_vinegar_core",
    13: "add_vinegar_post",
    14: "add_vinegar_prep",
    15: "background",
    16: "cut_cheese_core",
    17: "cut_cheese_post",
    18: "cut_cheese_prep",
    19: "cut_cucumber_core",
    20: "cut_cucumber_post",
    21: "cut_cucumber_prep",
    22: "cut_lettuce_core",
    23: "cut_lettuce_post",
    24: "cut_lettuce_prep",
    25: "cut_tomato_core",
    26: "cut_tomato_post",
    27: "cut_tomato_prep",
    28: "mix_dressing_core",
    29: "mix_dressing_post",
    30: "mix_dressing_prep",
    31: "mix_ingredients_core",
    32: "mix_ingredients_post",
    33: "mix_ingredients_prep",
    34: "peel_cucumber_core",
    35: "peel_cucumber_post",
    36: "peel_cucumber_prep",
    37: "place_cheese_into_bowl_core",
    38: "place_cheese_into_bowl_post",
    39: "place_cheese_into_bowl_prep",
    40: "place_cucumber_into_bowl_core",
    41: "place_cucumber_into_bowl_post",
    42: "place_cucumber_into_bowl_prep",
    43: "place_lettuce_into_bowl_core",
    44: "place_lettuce_into_bowl_post",
    45: "place_lettuce_into_bowl_prep",
    46: "place_tomato_into_bowl_core",
    47: "place_tomato_into_bowl_post",
    48: "place_tomato_into_bowl_prep",
    49: "serve_salad_onto_plate_core",
    50: "serve_salad_onto_plate_post",
    51: "serve_salad_onto_plate_prep"
}

custom_dict = {
    0: "add_dressing",
    1: "add_oil",
    2: "add_pepper",
    3: "background",
    4: "cut_into_pieces",
    5: "mix_dressing",
    6: "mix_ingredients",
    7: "peel_cucumber",
    8: "place_into_bowl",
    9: "serve_salad_onto_plate"
}

def from_chw_rgb_to_hwc_bgr(input_data):
    return np.moveaxis(input_data[::-1, :, :], 0, -1)


def from_chw_to_hwc(input_data):
    """
    Converts a numpy array from chw to hwc.

    :param input_data: Input array with channels first.
    :return: Output array with channels last.
    """
    return np.moveaxis(input_data, 0, -1)


def visualize_dataset(h5_dataset_path: os.path,
                      predict_npy_loc,
                      starting_frame):
    print(h5_dataset_path)

    with h5py.File(h5_dataset_path, 'r') as h5_file:
        rgbd = h5_file['rgbd']
        flow = h5_file['flow']
        print(flow.shape)
        mid_labels = h5_file['mid']
        fine_labels = h5_file['fine']
        coarse_labels = h5_file['coarse']
        custom_labels = h5_file['custom']

        for frame_idx in range(starting_frame, custom_labels.shape[0], 2):
            rgb_frame = np.squeeze(rgbd[frame_idx, :3, :, :])
            rgb_frame = from_chw_rgb_to_hwc_bgr(rgb_frame)

            depth_frame = np.squeeze(rgbd[frame_idx, 3, :, :])
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2BGR)

            print(flow[frame_idx].shape)

            flow_frame = np.squeeze(flow[frame_idx])
            flow_frame = from_chw_to_hwc(flow_frame)

            # print(mid_labels[frame_idx])
            # print(np.argmax(mid_labels[frame_idx]))
            mid_frame_label = mid_dict[np.argmax(mid_labels[frame_idx])]
            fine_frame_label = fine_dict[np.argmax(fine_labels[frame_idx])]
            coarse_frame_label = coarse_dict[np.argmax(coarse_labels[frame_idx])]
            custom_frame_label = custom_dict[np.argmax(custom_labels[frame_idx])]

            max_flow = 0.128
            flow_frame = np.clip(flow_frame, -max_flow, max_flow)
            flow_frame += max_flow
            flow_frame *= 255 * (1 / (2 * 0.128))
            flow_frame = flow_frame.astype(np.uint8)

            total_frame = np.zeros((240, 320, 3), np.uint8)

            total_frame[:120, :160] = rgb_frame
            total_frame[:120, 160:] = 2 * depth_frame
            total_frame[120:, :160] = flow_frame

            total_frame = cv2.resize(total_frame, dsize=None, fx=2.0, fy=2.0,
                                     interpolation=cv2.INTER_NEAREST)

            total_frame_pil = Image.fromarray(total_frame)
            draw = ImageDraw.Draw(total_frame_pil)
            font = ImageFont.truetype('NotoSans-Regular.ttf', size=25)

            draw.text((320, 240), fine_frame_label, font=font)
            draw.text((320, 300), mid_frame_label, font=font)
            draw.text((320, 360), custom_frame_label, font=font)
            draw.text((320, 420), coarse_frame_label, font=font)

            total_frame = np.array(total_frame_pil)

            cv2.imshow('Datasetvis', total_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualise the dataset")
    parser.add_argument("input", help="The input h5 file.", type=str)
    parser.add_argument("--frame",
                        help="The starting frame.",
                        default=0,
                        type=int)
    parser.add_argument("--predict_npy",
                        help="The predicted classes in npy file",
                        default=None,
                        type=str)
    args = parser.parse_args()

    input_loc = os.path.abspath(args.input)

    if not os.path.isfile(input_loc):
        print("The specified input file {} does not exist.".format(input_loc))
        sys.exit(1)

    predict_npy_loc = None
    if args.predict_npy is not None:
        predict_npy_loc = os.path.abspath(args.predict_npy)

        if not os.path.isfile(predict_npy_loc):
            print("The specified predict file {} "
                  "does not exist.".format(predict_npy_loc))
            sys.exit(1)

    try:
        visualize_dataset(input_loc, predict_npy_loc, args.frame)
    except ValueError as e:
        print("Error encountered: {}".format(e))
        sys.exit(1)
