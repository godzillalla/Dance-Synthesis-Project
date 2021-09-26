# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import yaml
import random
import numpy as np
from scipy.signal import argrelmin
from data_utils.skeleton import Skeleton
from transformations import quaternion_slerp
from data_utils.animation_data import get_origin_rotations, calculate_quaternions
from data_utils.quaternion_frame import convert_quaternion_frames_to_euler_frames
from reconstruction.bvh_writer import BVHWriter
from reconstruction.filter_rotations import DataFilter
from reconstruction.inverse_kinematics import InverseKinematics
from data_utils.constants import TEST_OUT_DIRECTORY, GLOBAL_INFO_DIRECTORY
from data_utils.utils import smooth_joints_around_transition_using_slerp, smooth_root_translation_around_transition, \
    quaternions_matrix


def get_bvh_info(file):
    f = open(GLOBAL_INFO_DIRECTORY + file, "r")
    bvh_info = yaml.load(f, Loader=yaml.Loader)
    info = dict()
    info["spine_index"] = bvh_info["spine"]
    info["left_hip_index"] = bvh_info["left_hip"]
    info["right_hip_index"] = bvh_info["right_hip"]
    info["fps"] = bvh_info["frame_per_second"]
    info["default_rotation_order"] = bvh_info["default_rotation_order"]
    return info


def up_sample_quaternion_frames(quaternion_frames, upsample=4):
    quaternion_all = [quaternion_frames[0]]
    for i, quaternions1 in enumerate(quaternion_frames):
        if i == 0:
            continue
        quaternions0 = quaternion_frames[i - 1]
        for add_frame in range(upsample - 1):
            time_factor = (add_frame + 1) / upsample
            add_quat = []
            for index in range(len(quaternions0)):
                quat0 = quaternions0[index]
                quat1 = quaternions1[index]
                add_quat.append(quaternion_slerp(quat0, quat1, time_factor))
            quaternion_all.append(add_quat)
        quaternion_all.append(quaternions1)
    return np.array(quaternion_all)


def up_sample_root_pos(root_pos, upsample=4):
    pos_all = [root_pos[0]]
    for i, pos1 in enumerate(root_pos):
        if i == 0:
            continue
        pos0 = root_pos[i - 1]
        for add_frame in range(upsample - 1):
            time_factor = (add_frame + 1) / upsample
            pos_all.append(pos0 + (pos1 - pos0) * time_factor)
        pos_all.append(pos1)
    return np.array(pos_all)


def quaternions_to_rotations(quaternions, root_pos, filter_window, up_sample):
    quaternion_all = up_sample_quaternion_frames(quaternions, up_sample)
    root_pos_all = up_sample_root_pos(root_pos, up_sample)
    quaternion_frames = []
    for i, quaternion in enumerate(quaternion_all):
        frame_data = [root_pos_all[i][0], root_pos_all[i][1], root_pos_all[i][2]]
        for j, q in enumerate(quaternion):
            frame_data.append(q[0])
            frame_data.append(q[1])
            frame_data.append(q[2])
            frame_data.append(q[3])
        quaternion_frames.append(frame_data)
    quaternion_frames = np.array(quaternion_frames)

    rotations = convert_quaternion_frames_to_euler_frames(quaternion_frames)

    rots = []
    for frame in rotations:
        rot = []
        for i in range(int(len(frame) / 3)):
            rot.append([frame[i * 3], frame[i * 3 + 1], frame[i * 3 + 2]])
        rots.append(rot)

    if filter_window > 1:
        data_filter = DataFilter(rots, filter_window)
        updated_rotations = data_filter.smooth_data()
        return updated_rotations
    return np.array(rots)


def save_to_bvh(rotations, skeleton, children_list, fps, out_file):
    writer = BVHWriter(rotations, out_file, skeleton.joint_names, children_list,
                       skeleton.offsets, fps)
    writer.write_to_bvh()
    print("Saved %d frames to %s" % (len(rotations), out_file))


def get_rotations(positions, skeleton, bvh_info, children_list):
    ik = InverseKinematics(positions, skeleton.joint_names, skeleton.parents, children_list, skeleton.offsets, bvh_info)
    rotations = ik.calculate_all_rotation()
    return np.array(rotations)


def get_feet_positions(frame_data, skeleton):
    root_pos = frame_data[:, :3]
    quaternions = frame_data[:, 3:]
    joint_num = int((frame_data.shape[1] - 3) / 4)
    # [frame_num, joint_num, 4]
    quaternions = quaternions.reshape(quaternions.shape[0], joint_num, 4)
    transforms = quaternions_matrix(quaternions)  # [..., J, 3, 3]
    local_positions = np.zeros(quaternions.shape[:-1] + (3,))  # [T, J, 3]
    for i, pi in enumerate(skeleton.parents):
        if pi == -1:
            continue
        local_positions[..., i, :] = np.matmul(transforms[..., pi, :, :],
                                               skeleton.offsets[i])
        local_positions[..., i, :] += local_positions[..., pi, :]
        transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                             transforms[..., i, :, :])

    root_pos = np.expand_dims(root_pos, 1)
    global_positions = local_positions + root_pos
    feet_pos = []
    lowest_y = []
    for frame in global_positions:
        point_cloud = []
        temp = frame[0, 1]
        for i, pos in enumerate(frame):
            if i in skeleton.left_fid or i in skeleton.right_fid:
                point_cloud.append(pos)
                temp = min(pos[1], temp)
        feet_pos.append(point_cloud)
        lowest_y.append(temp)
    return np.array(feet_pos), np.array(lowest_y)


def interpolate_root_height(positions):
    seq_num = positions.shape[0]
    start_pos = positions[0]
    end_pos = positions[seq_num - 1]
    for i in range(1, seq_num - 1):
        t = i / (seq_num - 1)
        positions[i] = start_pos + (end_pos - start_pos) * t


def smooth_key_frame_motion(frame_data, key_frame_idx, hwindow=8):
    joint_num = int((frame_data.shape[1] - 3) / 4)
    window = 2 * hwindow + 1
    for k in key_frame_idx:
        begin = max(0, k - hwindow)
        end = min(len(frame_data), k + hwindow + 1)
        win_frames = frame_data[begin:end]
        for j in range(joint_num):
            joint_param = list(range(3 + j * 4, 3 + j * 4 + 4))
            if k == key_frame_idx[0]:
                smooth_joints_around_transition_using_slerp(win_frames, joint_param, 0, window)
            else:
                smooth_joints_around_transition_using_slerp(win_frames, joint_param, hwindow, window)
        smooth_root_translation_around_transition(win_frames, hwindow, window)
        frame_data[begin:end] = win_frames


def plant_foot_on_ground(frame_data, key_frame_idx, skeleton):
    _, lowest_y = get_feet_positions(frame_data, skeleton)
    key_num = len(key_frame_idx)
    delta = 3.0
    ground_height = 0.0
    for k in key_frame_idx:
        ground_height += lowest_y[k] / key_num

    local_min_idx = np.array(argrelmin(lowest_y)[0])
    local_min_points = []
    for idx in local_min_idx:
        if lowest_y[idx] < ground_height - delta:
            local_min_points.append([idx, lowest_y[idx]])

    seq_num = len(frame_data)
    transitions = []
    for i, idx in enumerate(local_min_idx):
        if lowest_y[idx] < ground_height - delta:
            flag = 1
            for j in range(idx - 1, 0, -1):
                if lowest_y[j] >= ground_height - delta / 2:
                    transitions.append([j, idx])
                    flag = 0
                    break
                if i != 0 and j == local_min_idx[i - 1]:
                    flag = 0
                    if [j, idx] not in transitions:
                        transitions.append([j, idx])
                    break
            k = 0
            if flag:
                k = 1
            flag = 1
            for j in range(idx + 1, seq_num):
                if (i != len(local_min_idx) - 1 and j == local_min_idx[i + 1]) or \
                        lowest_y[j] >= ground_height - delta / 2:
                    if k:
                        transitions.append([0, j])
                    else:
                        transitions.append([idx, j])
                    flag = 0
                    break
            if flag:
                for j in range(len(transitions) - 1, 0, -1):
                    if transitions[j][1] == idx:
                        temp = transitions[j][0]
                        transitions.remove(transitions[j])
                        transitions.append([temp, seq_num - 1])
                        break
            frame_data[idx, 1] += ground_height - delta + random.uniform(-1.0, 1.0) - lowest_y[idx]

    for i in range(len(transitions)):
        start = transitions[i][0]
        end = transitions[i][1]
        positions = frame_data[start:end + 1, :3]
        interpolate_root_height(positions)
        frame_data[start:end + 1, :3] = positions

    _, new_lowest_y = get_feet_positions(frame_data, skeleton)


def save_test_result_to_bvh(changed_positions, root_pos, root_rot, key_idx, out_file,
                            bvh_info="bvh_cyprus.yml", filter_win=1, upsample=1):
    skeleton = Skeleton()
    bvh_info = get_bvh_info(bvh_info)
    skeleton.offsets[0] = [0, 0, 0]
    children_list = dict()
    for i in range(len(skeleton.joint_names)):
        children = []
        for j in range(len(skeleton.parents)):
            if i == skeleton.parents[j]:
                children.append(j)
        children_list[i] = children

    rotations = get_rotations(changed_positions, skeleton, bvh_info, children_list)
    quaternions = calculate_quaternions(rotations, bvh_info["default_rotation_order"])
    # [frame_num, joint_num, 4]
    quaternions = get_origin_rotations(quaternions[:, 1:], root_rot)

    joint_num = quaternions.shape[1]
    quaternions.resize((quaternions.shape[0], quaternions.shape[1] * quaternions.shape[2]))
    frame_quaternions = np.concatenate((root_pos, quaternions), axis=-1)
    smooth_key_frame_motion(frame_quaternions, key_idx)
    plant_foot_on_ground(frame_quaternions, key_idx, skeleton)
    origin_quaternions = frame_quaternions[:, 3:].reshape(frame_quaternions.shape[0], joint_num, 4)
    origin_rotations = quaternions_to_rotations(origin_quaternions, frame_quaternions[:, :3], filter_win, upsample)

    # origin_rotations = quaternions_to_rotations(quaternions, root_pos, filter_win, upsample)
    save_to_bvh(origin_rotations, skeleton,
                children_list, bvh_info["fps"], TEST_OUT_DIRECTORY + out_file)
