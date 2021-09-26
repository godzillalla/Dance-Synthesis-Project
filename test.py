# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import os
import time
import yaml
import math
import random
import argparse
import numpy as np
from config import Config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from prediction_network import test_prediction, divide_data, DanceDataset
from reconstruction.get_result import save_test_result_to_bvh
from data_utils.animation_data import get_vel_factor
from data_utils.constants import GLOBAL_INFO_DIRECTORY, TEST_OUT_DIRECTORY

random_select = True
static_begin_frame = 380
static_test_file = 115
test_interval = [60, 90, 50, 70]
vel_factor_change_ratio = 1.0

# change velocity factor
vel_factor_choose = False
vel_factor_begin_frame = 60
vel_factor_file = 5

# change first key frame data
first_keyframe_choose = False
first_keyframe_frame = 246
first_keyframe_file = 116

# change last key frame data
last_keyframe_choose = False
last_keyframe_frame = 226  # 66
last_keyframe_file = 121  # 122

# change all keyframe pose
all_keyframe_choose = False
all_keyframe_begin_frame = 309
all_keyframe_file = 101


def get_npz_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.npy')]


def load_data(data_path):
    """
    data(npy): [frame_num, 215]  215 = 23 * 3 + 3 + 1 + 4 + 23 * 3 + 23 * 3
    include:
        joint_pos        # [joint_num, 3] - [23, 3]
        root_pos         # [3]
        root_rot         # [1]
        contact          # [4]
        velocity         # [joint_num, 3] - [23, 3]
        acceleration     # [joint_num, 3] - [23, 3]
    ps:
        joint_num: 24
    """
    all_files = get_npz_files(data_path)
    # train_data_num = int(len(all_files) * train_data_proportion)
    data_set = []
    data_name = []
    file_num = len(all_files)
    for i, bvh in enumerate(all_files):
        strs = bvh.split("\\")
        data_name.append(strs[-1][:-4])
        data = np.load(bvh)
        print("load file %s (%d/%d)" % (strs[-1][:-4], i, file_num))
        print("  shape:", data.shape)
        data_set.append(data[:, :-72])
    print()
    print("data file num:", len(data_name), len(data_set))
    return data_set, data_name


def position_vector(pos_list):
    """
    :param pos_list: [T, (J - 1) * 3]
    :return: positions: [T, J, 3]
    """
    positions = []
    for frame_pos in pos_list:
        pos = [[0.0, 0.0, 0.0]]
        for i in range(int(len(frame_pos) / 3)):
            pos.append([frame_pos[i * 3], frame_pos[i * 3 + 1], frame_pos[i * 3 + 2]])
        positions.append(pos)
    return np.array(positions)


def batch_position_vector(pos_list):
    """
    :param pos_list: [B, T, (J - 1) * 3]
    :return: positions: [B, T, J, 3]
    """
    batch_size = pos_list.shape[0]
    seq_len = pos_list.shape[1]
    joint_num = int(pos_list.shape[2] / 3 + 1)
    positions = np.zeros((batch_size, seq_len, joint_num, 3))
    for i in range(1, joint_num):
        positions[:, :, i, :] = pos_list[:, :, i * 3 - 3:i * 3]
    return positions


def draw_root_trajectory(pred_root_pos, true_root_pos, test_section, time_str):
    pred_pos_y = pred_root_pos[:, 1]
    pred_pos_x = pred_root_pos[:, 0]
    pred_pos_z = pred_root_pos[:, 2]

    true_pos_y = true_root_pos[:, 1]
    true_pos_x = true_root_pos[:, 0]
    true_pos_z = true_root_pos[:, 2]

    plt.plot(pred_pos_x, pred_pos_z, color='green', label='pred xz')
    plt.scatter(pred_pos_x, pred_pos_z)

    plt.plot(true_pos_x, true_pos_z, color='red', label='true xz')
    plt.scatter(true_pos_x, true_pos_z)
    plt.legend()

    plt.savefig('test_out/root_xz_' + time_str + '_' + '.jpg')
    plt.close()

    x = np.arange(0, pred_root_pos.shape[0], 1)
    plt.plot(x, pred_pos_y, color='green', label='pred y')
    plt.scatter(x, pred_pos_y)

    plt.plot(x, true_pos_y, color='red', label='true xz')
    plt.scatter(x, true_pos_y)
    plt.legend()
    plt.savefig('test_out/root_y_' + time_str + '_' + '.jpg')
    plt.close()


def draw_vel_factor(positions, true_vel_factor, test_section, time_str):
    def draw_one_vel_factor(pre_vel, true_vel, index):
        x = np.arange(0, pre_vel.shape[0], 1)
        plt.plot(x, pre_vel, color='green', label='pred y')
        plt.scatter(x, pre_vel)

        plt.plot(x, true_vel, color='red', label='true xz')
        plt.scatter(x, true_vel)
        plt.legend()
        plt.savefig('test_out/vel_' + time_str + '_' + str(index) + '.jpg')
        plt.close()

    temp_positions = np.concatenate([[positions[0]], positions], axis=0)  # [T + 1, J, 3]
    velocity = temp_positions[1:] - temp_positions[:-1]  # [T, J, 3]

    pred_vel_factor = get_vel_factor(velocity)

    for i in range(pred_vel_factor.shape[1]):
        draw_one_vel_factor(pred_vel_factor[:, i], true_vel_factor[:, i], i)


def generate_test_data(raw_data, config, test_section):
    frame_num = len(raw_data)
    interval = test_section["interval"]
    frame_sum = 1

    mask = [1]
    for i in interval:
        frame_sum += i + 1
        for j in range(i):
            mask.append(0)
        mask.append(1)

    if frame_sum + 10 >= frame_num:
        print("not enough frames(%d - %d)" % (frame_sum, frame_num))
    if random_select:
        begin_frame = random.randint(0, frame_num - frame_sum - 10)
    else:
        begin_frame = static_begin_frame

    test_section["begin_frame"] = begin_frame
    print("choose test data, test_file: %d, begin_frame: %d" % (test_section["test_file"], begin_frame))
    test_data = []
    target = []
    key_frames = []
    gt_info = []
    time_label = []
    vel_factor = []
    vel_loc = config.state_encoder_input_size + config.derivative_encoder_input_size
    for i in range(frame_sum):
        gt_info.append(raw_data[begin_frame + i])
        time_label.append(raw_data[begin_frame + i][config.pos_dim:config.pos_dim + config.root_pos_dim])
        vel_factor.append(raw_data[begin_frame + i][vel_loc:vel_loc + config.vel_factor_dim])
        if mask[i] == 1:
            key_frames.append(i)
            test_data.append(raw_data[begin_frame + i])
            if i != 0:
                target.append(raw_data[begin_frame + i][:config.target_encoder_input_size])
        else:
            test_data.append(np.zeros(raw_data[0].shape))
    test_section["key_frame"] = key_frames

    target = np.expand_dims(np.array(target), axis=0)
    test_data = np.expand_dims(np.array(test_data), axis=0)
    time_label = np.expand_dims(np.array(time_label), axis=0)
    vel_factor = np.expand_dims(np.array(vel_factor), axis=0)
    print("    choose frame %d - %d for test" % (begin_frame, begin_frame + frame_sum))
    return np.array(mask), time_label, vel_factor, test_data, target, np.array(gt_info)


def generate_different_vel_factor(raw_data, config, test_section):
    frame_num = len(raw_data)
    interval = test_section["interval"]
    frame_sum = 1

    mask = [1]
    for i in interval:
        frame_sum += i + 1
        for j in range(i):
            mask.append(0)
        mask.append(1)

    if frame_sum + 10 >= frame_num:
        print("not enough frames(%d - %d)" % (frame_sum, frame_num))

    begin_frame = test_section["vel_factor_begin_frame"]

    print("choose vel_factor, test_file: %d, begin_frame: %d" % (test_section["vel_factor_file"], begin_frame))
    vel_factor = []
    vel_loc = config.state_encoder_input_size + config.derivative_encoder_input_size
    for i in range(frame_sum):
        vel_factor.append(raw_data[begin_frame + i][vel_loc:vel_loc + config.vel_factor_dim])
    vel_factor = np.expand_dims(np.array(vel_factor), axis=0)
    return vel_factor


def change_first_key_frame(raw_data, choose_frame, test_data, time_label, target, key_idx, mean, std, config):
    key_frame_data = raw_data[choose_frame]

    changed_root_pos = key_frame_data[config.pos_dim:config.pos_dim + config.root_pos_dim] * \
                       std[config.pos_dim:config.pos_dim + config.root_pos_dim] + \
                       mean[config.pos_dim:config.pos_dim + config.root_pos_dim]

    # (1, 273, 3)
    test_time_label = time_label * \
                      std[config.pos_dim:config.pos_dim + config.root_pos_dim] + \
                      mean[config.pos_dim:config.pos_dim + config.root_pos_dim]
    delta_root_pos = changed_root_pos - test_time_label[0, 0, :]

    test_time_label = test_time_label + delta_root_pos

    time_label[:, :, :] = (test_time_label - mean[config.pos_dim:config.pos_dim + config.root_pos_dim]) / \
                          std[config.pos_dim:config.pos_dim + config.root_pos_dim]
    test_data[0, 0, :config.pos_dim + config.root_pos_dim + config.root_rot_dim] = \
        key_frame_data[:config.pos_dim + config.root_pos_dim + config.root_rot_dim]
    test_data[:, :, config.pos_dim:config.pos_dim + config.root_pos_dim] = time_label

    for i in range(1, len(key_idx)):
        target[:, i - 1, config.pos_dim:config.pos_dim + config.root_pos_dim] = time_label[:, key_idx[i], :]


def change_all_key_frame(raw_data, begin_frame, test_data, target, mask, config):
    target_idx = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            test_data[0, i, :config.pos_dim] = raw_data[begin_frame + i][:config.pos_dim]
            if i != 0:
                target[0, target_idx, :config.pos_dim] = raw_data[begin_frame + i][:config.pos_dim]
                target_idx += 1


def generate_test_data_fix_window(data, config):
    vel_loc = config.state_encoder_input_size + config.derivative_encoder_input_size
    # batch_size = data.shape[0]
    frame_num = data.shape[1]
    gt_info = data.copy()

    time_label = data[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
    vel_factor = data[..., vel_loc:vel_loc + config.vel_factor_dim]
    test_data = data.copy()

    target = data[:, -1:, :config.target_encoder_input_size]
    for i in range(frame_num):
        if i != 0 and i != frame_num - 1:
            test_data[:, i, :] = np.zeros(test_data[:, i, :].shape)

    return time_label, vel_factor, test_data, target, gt_info


def getL2Q(true_positions, test_positions):
    # positions [B, T, 24 * 3]
    batch_size = true_positions.shape[0]
    seq_len = true_positions.shape[1]
    z = np.sum(np.linalg.norm(true_positions - test_positions, axis=-1)) / seq_len / batch_size
    return z


def get_vel_factor_for_batch(positions, vel_factor_dim):
    # [B, T, 24 * 3]
    batch_size = positions.shape[0]
    seq_len = positions.shape[1]
    joint_num = int(positions.shape[2] / 3)

    temp_positions = np.concatenate([positions[:, :1], positions], axis=1)
    velocity = temp_positions[:, 1:] - temp_positions[:, :-1]

    weight = [1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    parts = [0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0]
    weight_sum = []
    for part in range(5):
        p_sum = 0
        for j in range(joint_num):
            if parts[j] == part:
                p_sum += weight[j]
        weight_sum.append(p_sum)

    vel_factor = np.zeros((batch_size, seq_len, vel_factor_dim))
    for part in range(vel_factor_dim):
        for j in range(joint_num):
            if parts[j] == part:
                vel_factor[..., part] += weight[j] / weight_sum[part] * \
                                         pow(pow(velocity[..., j * 3], 2) +
                                             pow(velocity[..., j * 3 + 1], 2) +
                                             pow(velocity[..., j * 3 + 2], 2), 0.5)
    return vel_factor


def evaluation_prediction(args, win=50):
    config = Config()
    data_set, data_name = load_data(args.data_path)
    win_step = math.ceil(config.win_step_factor * win)
    mean_std_data = np.load(GLOBAL_INFO_DIRECTORY + "mean_std.npz")
    mean, std = mean_std_data["mean"], mean_std_data["std"]  # 215, 215

    test_data = divide_data(data_set, win, win_step)
    test_loader = DataLoader(DanceDataset(test_data), batch_size=config.batch_size)
    mask = [1]
    for j in range(win - 2):
        mask.append(0)
    mask.append(1)
    mask = np.array(mask)

    predict_model_path = config.model_dir + args.predict_model_path

    position_eval = 0.0
    vel_factor_eval = 0.0
    root_trajectory_eval = 0.0
    v_delta = 1.0
    r_delta = 7.0
    num = 0
    for i, _data in enumerate(test_loader):
        print("evaluation %d/%d" % (i, len(test_loader)))
        data = _data.numpy()
        time_factor, vel_factor, test_data, target, gt_seq = generate_test_data_fix_window(data, config)
        print("input data:", test_data.shape)
        print("time_factor:", time_factor.shape)
        predict_seq = test_prediction(mask, time_factor, vel_factor, test_data, target, predict_model_path)
        predict_seq = predict_seq * std[:-(config.velocity_dim + config.vel_factor_dim)] + \
                      mean[:-(config.velocity_dim + config.vel_factor_dim)]
        # [B, T, 23 * 3]

        pred_positions = np.zeros((predict_seq.shape[0], predict_seq.shape[1], config.pos_dim + config.root_pos_dim))
        pred_positions[..., config.root_pos_dim:] = predict_seq[..., :config.pos_dim]
        pred_positions[..., :config.root_pos_dim] = predict_seq[...,
                                                    config.pos_dim:config.pos_dim + config.root_pos_dim]
        gt_seq = gt_seq[..., :-config.vel_factor_dim] * std[:-(config.velocity_dim + config.vel_factor_dim)] + \
                 mean[:-(config.velocity_dim + config.vel_factor_dim)]
        gt_positions = np.zeros((gt_seq.shape[0], gt_seq.shape[1], config.pos_dim + config.root_pos_dim))
        gt_positions[..., config.root_pos_dim:] = gt_seq[..., :config.pos_dim]
        gt_positions[..., :config.root_pos_dim] = gt_seq[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
        position_eval += getL2Q(gt_positions, pred_positions)

        pred_vel_factor = get_vel_factor_for_batch(pred_positions, config.vel_factor_dim)
        gt_vel_factor = vel_factor * std[-(config.vel_factor_dim + config.velocity_dim):-config.velocity_dim] + \
                        mean[-(config.vel_factor_dim + config.velocity_dim):-config.velocity_dim]
        gt_vel_factor[:, 0] = 0.0
        vel_factor_delta = gt_vel_factor - pred_vel_factor
        v_sum = vel_factor_delta.shape[0] * vel_factor_delta.shape[1] * vel_factor_delta.shape[2]
        v = 0
        for item in vel_factor_delta.flatten():
            if -v_delta < item < v_delta:
                v += 1
        vel_factor_eval += v / v_sum

        gt_root_trajectory = gt_positions[..., :config.root_pos_dim]
        pred_root_trajectory = predict_seq[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
        root_trajectory_delta = gt_root_trajectory - pred_root_trajectory
        r_sum = root_trajectory_delta.shape[0] * root_trajectory_delta.shape[1]
        root_trajectory_delta = root_trajectory_delta.reshape(r_sum, 3)
        r = 0
        for item in root_trajectory_delta:
            dis = pow(item[0] * item[0] + item[1] * item[1] + item[2] * item[2], 0.5)
            if -r_delta < dis < r_delta:
                r += 1
        root_trajectory_eval += r / r_sum
        num += 1
        '''
        # ========================================================================
        # save file
        random_num = random.randint(0, 127)
        predict_seq = predict_seq[random_num]
        predict_seq = predict_seq * std[:-(config.velocity_dim + config.vel_factor_dim)] + \
                      mean[:-(config.velocity_dim + config.vel_factor_dim)]
        pred_positions = position_vector(predict_seq[..., :config.pos_dim])
        pred_root_pos = predict_seq[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
        pred_root_rot = predict_seq[...,
                        config.pos_dim + config.root_pos_dim: config.pos_dim + config.root_pos_dim + config.root_rot_dim]
        pred_file_name = str(i) + '_' + str(random_num) + "_pred_result.bvh"
        save_test_result_to_bvh(pred_positions, pred_root_pos, pred_root_rot, pred_file_name)
        gt_seq = gt_seq[random_num]
        gt_seq = gt_seq * std[:-(config.velocity_dim + config.vel_factor_dim)] + \
                 mean[:-(config.velocity_dim + config.vel_factor_dim)]
        gt_positions = position_vector(gt_seq[..., :config.pos_dim])
        gt_root_pos = gt_seq[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
        gt_root_rot = gt_seq[...,
                      config.pos_dim + config.root_pos_dim: config.pos_dim + config.root_pos_dim + config.root_rot_dim]
        gt_file_name = str(i) + '_' + str(random_num) + "_gt_result.bvh"
        save_test_result_to_bvh(gt_positions, gt_root_pos, gt_root_rot, gt_file_name)
        # ========================================================================
        '''
        print()
    position_eval = position_eval / num
    vel_factor_eval = vel_factor_eval / num
    root_trajectory_eval = root_trajectory_eval / num
    print("position evaluation: %f (of %d batches)" % (position_eval, num))
    print("velocity factor evaluation: %f (of %d batches)" % (vel_factor_eval, num))
    print("root trajectory evalution: %f (of %d batches)" % (root_trajectory_eval, num))


def test_prediction_network(args):
    config = Config()
    mean_std_data = np.load(GLOBAL_INFO_DIRECTORY + "mean_std.npz")
    mean, std = mean_std_data["mean"], mean_std_data["std"]  # 215, 215

    data_set, data_name = load_data(args.data_path)
    data_num = len(data_name)
    test_section = {"test_file": 0, "interval": [], "key_frame": [], "begin_frame": 0}
    if random_select:
        # [a, b]
        train_num = int(data_num * config.train_data_proportion)
        test_file = random.randint(train_num, data_num - 1)
    else:
        test_file = static_test_file
    test_section["test_file"] = test_file
    test_section["test_file_name"] = data_name[test_file]
    if test_file < int(data_num * config.train_data_proportion):
        use_train_data = True
    else:
        use_train_data = False

    test_section["interval"] = test_interval
    test_section["use_train_data"] = use_train_data

    # choose data
    print("Choose file %s (%d/%d) for testing." % (data_name[test_file], test_file, data_num))
    data = data_set[test_file]
    mask, time_factor, vel_factor, test_data, target, gt_seq = generate_test_data(data, config, test_section)
    test_section["vel_factor_begin_frame"] = test_section["begin_frame"]
    test_section["frame_sum"] = test_data.shape[1]

    test_section["vel_factor_choose"] = vel_factor_choose
    if vel_factor_choose:
        test_section["vel_factor_file"] = vel_factor_file
        test_section["vel_factor_begin_frame"] = vel_factor_begin_frame
        vel_factor = generate_different_vel_factor(data_set[vel_factor_file], config, test_section)

    test_section["first_keyframe_choose"] = first_keyframe_choose
    if first_keyframe_choose:
        test_section["first_keyframe_file"] = first_keyframe_file
        test_section["first_keyframe_frame"] = first_keyframe_frame
        print("choose first key frame, file: %d, frame: %d" % (first_keyframe_file, first_keyframe_frame))
        change_first_key_frame(data_set[first_keyframe_file], first_keyframe_frame, test_data,
                               time_factor, target, test_section["key_frame"], mean, std, config)

    test_section["last_keyframe_choose"] = last_keyframe_choose
    if last_keyframe_choose:
        test_section["last_keyframe_file"] = last_keyframe_file
        test_section["last_keyframe_frame"] = last_keyframe_frame
        print("choose last key frame, file: %d, frame: %d" % (last_keyframe_file, last_keyframe_frame))
        key_frame_data = data_set[last_keyframe_file][last_keyframe_frame]
        print("last_key_frame_data:", key_frame_data.shape)
        test_data[0, -1, :config.pos_dim] = key_frame_data[:config.pos_dim]
        loc = config.pos_dim + config.root_pos_dim
        test_data[0, -1, loc:loc + config.root_rot_dim] = key_frame_data[loc:loc + config.root_rot_dim]
        target[0, -1, :config.pos_dim] = key_frame_data[:config.pos_dim]
        target[0, -1, loc:loc + config.root_rot_dim] = key_frame_data[loc:loc + config.root_rot_dim]

    test_section["all_keyframe_choose"] = all_keyframe_choose
    if all_keyframe_choose:
        test_section["all_keyframe_file"] = all_keyframe_file
        test_section["all_keyframe_begin_frame"] = all_keyframe_begin_frame
        print("choose all key frame, file: %d, frame: %d" % (all_keyframe_file, all_keyframe_begin_frame))
        change_all_key_frame(data_set[all_keyframe_file], all_keyframe_begin_frame, test_data, target, mask, config)

    now_time = time.strftime("%d-%H-%M", time.localtime())

    test_section["vel_factor_change_ratio"] = vel_factor_change_ratio

    true_vel_factor = vel_factor * std[-config.vel_factor_dim - 72:-72] + mean[-config.vel_factor_dim - 72:-72]

    vel_factor[:, :, 0] = (true_vel_factor[:, :, 0] * 1.0 - mean[-config.vel_factor_dim - 72]) / \
                          std[-config.vel_factor_dim - 72]
    vel_factor[:, :, 1] = (true_vel_factor[:, :, 1] * 1.5 - mean[-config.vel_factor_dim - 72 + 1]) / \
                          std[-config.vel_factor_dim - 72 + 1]
    vel_factor[:, :, 2] = (true_vel_factor[:, :, 2] * 0.5 - mean[-config.vel_factor_dim - 72 + 2]) / \
                          std[-config.vel_factor_dim - 72 + 2]

    vel_factor[:, :130, 3] = (true_vel_factor[:, :130, 3] * 2.0 - mean[-config.vel_factor_dim - 72 + 3]) / \
                             std[-config.vel_factor_dim - 72 + 3]
    vel_factor[:, 130:, 3] = (true_vel_factor[:, 130:, 3] * 1.5 - mean[-config.vel_factor_dim - 72 + 3]) / \
                             std[-config.vel_factor_dim - 72 + 3]
    vel_factor[:, :, 4] = (true_vel_factor[:, :, 4] * 0.5 - mean[-config.vel_factor_dim - 72 + 4]) / \
                          std[-config.vel_factor_dim - 72 + 4]

    # test network
    predict_model_path = config.model_dir + args.predict_model_path
    predict_seq = test_prediction(mask, time_factor, vel_factor, test_data, target, predict_model_path)[0]
    predict_seq = predict_seq * std[:-(config.velocity_dim + config.vel_factor_dim)] + \
                  mean[:-(config.velocity_dim + config.vel_factor_dim)]

    filename = TEST_OUT_DIRECTORY + now_time + "_pn_test_parameter" + ".yml"
    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(test_section, f)
    print("Save test parameter to " + filename, "\n")
    bvh_file_name = now_time + "_pn_result.bvh"

    changed_positions = position_vector(predict_seq[..., :config.pos_dim])
    root_pos = predict_seq[..., config.pos_dim:config.pos_dim + config.root_pos_dim]
    root_rot = predict_seq[...,
               config.pos_dim + config.root_pos_dim: config.pos_dim + config.root_pos_dim + config.root_rot_dim]
    true_root_pos = time_factor[0] * \
                    std[config.pos_dim:config.pos_dim + config.root_pos_dim] + \
                    mean[config.pos_dim:config.pos_dim + config.root_pos_dim]

    vel_loc = config.state_encoder_input_size + config.derivative_encoder_input_size
    true_vel_factor = vel_factor[0] * std[vel_loc:vel_loc + 1] + mean[vel_loc:vel_loc + 1]
    pos = changed_positions.copy()
    pos[:, 0] = root_pos
    # draw_root_trajectory(root_pos, true_root_pos, test_section, now_time)
    # draw_vel_factor(pos, true_vel_factor, test_section, now_time)

    save_test_result_to_bvh(changed_positions, root_pos, root_rot, test_section['key_frame'], bvh_file_name)


def parse_args():
    parser = argparse.ArgumentParser("test")
    parser.add_argument("--test", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--predict_model_path", type=str, default="")
    return parser.parse_args()


# --test prediction --data_path data/Cyprus_out/ --predict_model_path 2021.04.14/
# --test prediction --data_path data/Cyprus_out/
if __name__ == '__main__':
    args = parse_args()
    if args.test == "prediction":
        print("Test Prediction Network!")
        # evaluation_prediction(args, 150)
        test_prediction_network(args)
