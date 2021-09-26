# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import numpy as np
from data_utils.quaternion_frame import euler_to_quaternion, check_quat, quaternion_to_euler

class DataFilter(object):
    def __init__(self, rotations, window_size):
        self.rotations = rotations
        self.quaternions = []
        self.frame_num = len(rotations)
        self.win_size = window_size
        self.updated_rotations = []

    def smooth_data(self):
        self.euler_to_quaternions()
        for frame in range(self.frame_num):
            smooth_list = self.get_smooth_list(frame, self.win_size)
            rot_list = self.smooth_one_frame(smooth_list)
            euler_list = []
            for i, data in enumerate(rot_list):
                if i == 0:
                    euler_list.append(data)
                    continue
                euler = quaternion_to_euler(data)
                euler_list.append(euler)
            self.updated_rotations.append(euler_list)
        return self.updated_rotations

    def smooth_one_frame(self, smooth_list):
        rot_list = []
        channel_num = len(smooth_list[0])
        for j in range(channel_num):
            if j == 0:
                rot = np.array([0.0, 0.0, 0.0])
            else:
                rot = np.array([0.0, 0.0, 0.0, 0.0])
            # rot = 0.0
            # print("rot:", rot, smooth_list[0][j])
            for i in range(len(smooth_list)):
                rot += smooth_list[i][j]
            rot_list.append(rot / len(smooth_list))
        return rot_list

    def get_smooth_list(self, frame, window):
        start = frame - int(window / 2)
        final = frame + int(window / 2)
        if start < 0:
            start = 0
        if final >= self.frame_num:
            final = self.frame_num - 1
        smooth_list = []
        for i in range(start, final):
            smooth_list.append(self.quaternions[i])
        return smooth_list

    def euler_to_quaternions(self):
        for frame_list in self.rotations:
            quats = []
            for i, euler, in enumerate(frame_list):
                if i == 0:
                    quats.append(euler)
                    continue
                quat = euler_to_quaternion(euler)
                quats.append(quat)
            self.quaternions.append(quats)

        for frame in range(len(self.quaternions)):
            if frame == 0:
                continue
            for i in range(len(self.quaternions[frame])):
                self.quaternions[frame][i] = check_quat(self.quaternions[frame][i], self.quaternions[frame - 1][i])




