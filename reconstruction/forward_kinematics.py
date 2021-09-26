# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import numpy as np
import sys
from transformations import quaternion_matrix
from data_utils.quaternion_frame import euler_to_quaternion

ORIGIN = np.array([0, 0, 0, 1])


class ForwardKinematics(object):
    def __init__(self, joints_name, init_offset):
        # print("Initialize class ForwardKinematics.")
        self.global_matrix = dict()
        self.joints = joints_name
        self.init_offset = init_offset

    def clear_matrix(self):
        self.global_matrix.clear()

    def get_node_position(self, node_index, node_rotation, parent_index=0):
        '''
        :param node_index: Index of current node
        :param node_rotation: The rotation of current node
        :param parent_index: Index of parent node
        :return: The global coordinates of current node
        '''
        if node_index == 0:
            return self.global_matrix[self.joints[node_index]][:3, 3]
        global_matrix = self.get_global_matrix(parent_index, node_index, node_rotation)
        point = np.dot(global_matrix, ORIGIN)
        return point

    def get_global_matrix(self, parent_index, node_index, rotation):
        parent_name = self.joints[parent_index]
        if parent_name not in self.global_matrix:
            print("Error: The parent node[", parent_name, "] should be computed first according to the tree structure!")
            sys.exit()
        node_name = self.joints[node_index]
        offset = self.init_offset[node_index]
        self.global_matrix[node_name] = np.dot(self.global_matrix[parent_name], self.get_local_matrix(rotation, offset))
        return self.global_matrix[node_name]

    def get_local_matrix(self, rotation, offset):
        quaternion = euler_to_quaternion(rotation)
        m = quaternion_matrix(quaternion)
        m[:3, 3] = offset
        return m

    def set_root_matrix(self, position, rotation, index, name):
        quaternion = euler_to_quaternion(rotation)
        root_matrix = quaternion_matrix(quaternion)
        root_matrix[:3, 3] = [t + o for t, o in zip(position, self.init_offset[index])]
        self.global_matrix[name] = root_matrix
