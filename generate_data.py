# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import os
import argparse
import numpy as np
import yaml
from data_utils.skeleton import Skeleton
from data_utils.animation_data import AnimaitonData
from data_utils.constants import DATA_DIRECTORY, GLOBAL_INFO_DIRECTORY, SAVE_YAML_FILE


def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh')]


def get_data_from_bvh(filename, downsample, skeleton):
    anim = AnimaitonData(filename=filename, downsample=downsample, skeleton=skeleton)
    # rot, quaternion, joint_pos, changed_quaternion, changed_pos,
    # root_pos, root_rot, contact, velocity, acceleration
    data = anim.get_down_sample_data()

    frame_num = len(data.root_rot)
    changed_pos = data.changed_pos.reshape(frame_num, -1)
    velocity = data.velocity.reshape(frame_num, -1)
    acceleration = data.acceleration.reshape(frame_num, -1)
    vel_factor = data.vel_factor

    # [frame_num, 215]  215 = 23 * 3 + 3 + 1 + 4 + 24 * 3 + 1 + 24 * 3
    return np.concatenate([changed_pos, data.root_pos, data.root_rot,
                           data.contact, velocity, vel_factor, acceleration], axis=-1)


def save_parent_and_bone(filename, parents, bone_length):
    # save
    info = {"parents": parents, "bone_length": bone_length}
    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(info, f)


def generate_database_cyprus(bvh_path, output_path, downsample, separate=True):
    bvh_files = get_bvh_files(bvh_path)
    skeleton = Skeleton()
    file_num = len(bvh_files)
    save_parent_and_bone(GLOBAL_INFO_DIRECTORY + SAVE_YAML_FILE, skeleton.parents, skeleton.bone_length)

    data_set = []
    data_frame = []
    for i, bvh in enumerate(bvh_files):
        print()
        print("Processing file %s (%i/%i)" % (bvh, i + 1, file_num))
        unit = get_data_from_bvh(bvh, downsample, skeleton)  # [frame_num, frame_data]   frame_data - 215
        data_frame.append(len(unit))
        for frame in unit:
            data_set.append(frame)
    data_set = np.array(data_set)  # (sum_frame_num, frame_data)

    mean = np.mean(data_set, axis=0)
    std = np.std(data_set, axis=0)
    std[np.where(std == 0)] = 1e-9
    mean_std_file = GLOBAL_INFO_DIRECTORY + "mean_std.npz"
    np.savez(mean_std_file, mean=mean, std=std)
    print("saved mean and std of data to", mean_std_file)
    norm_data = (data_set - mean) / std

    final_data = []
    index = 0
    for frame_num in data_frame:
        unit = []
        for i in range(frame_num):
            unit.append(norm_data[index])
            index += 1
        final_data.append(np.array(unit))
    final_data = np.array(final_data, dtype=object)  # (file_num, frame_num_in_file, frame_data)
    print("final_data:", final_data.shape)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if separate:  # [frame_num, 215]
        data_name = []
        for i, bvh in enumerate(bvh_files):
            strs = bvh.split('/')
            name = str(i + 1) + "_" + strs[-1][:-4]
            data_name.append(name)
        out_num = len(final_data)
        for i, data in enumerate(final_data):
            print()
            print("saving file %s (%i/%i) to %s" % (data_name[i], i + 1, out_num, output_path + data_name[i]))
            print("  shape:", data.shape)
            np.save(output_path + data_name[i], data)
    else:  # [bvh_file_num, frame_num, frame_data] frame_data - 215
        print()
        print("saving a final file (1/1)")
        print("  shape:", final_data.shape)
        for data in final_data:
            print("    ", data.shape)
        np.save(output_path + "Dance_Sum", final_data)


def parse_args():
    parser = argparse.ArgumentParser("generate_data")
    parser.add_argument("--dataset", type=str, default="Cyprus")
    parser.add_argument("--downsample", type=int, default=1)
    return parser.parse_args()


# --dataset Cyprus --downsample 1
if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "Cyprus":
        input_path = DATA_DIRECTORY + args.dataset
        output_path = DATA_DIRECTORY + args.dataset + "_out" + "/"
        generate_database_cyprus(input_path, output_path, args.downsample)
