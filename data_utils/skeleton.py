import yaml
import os
from .BVH_reader import BVHReader
from .constants import SKELETON_FILE
from .utils import get_distance_3d_point


def calculate_bone_length_from_position(positions, parents):
    bone_length = []
    for joint, parent in enumerate(parents):
        if parent == -1:
            continue
        joint_pos = positions[joint]
        parent_pos = positions[parent]
        length = get_distance_3d_point(joint_pos, parent_pos)
        bone_length.append(length)
    return bone_length


def calculate_bone_length_from_offsets(offsets, parents):
    bone_length = []
    for joint, parent in enumerate(parents):
        if parent == -1:
            continue
        joint_offset = offsets[joint]
        parent_pos = [0, 0, 0]
        length = get_distance_3d_point(joint_offset, parent_pos)
        bone_length.append(length)
    return bone_length


class Skeleton:
    def __init__(self):
        f = open(SKELETON_FILE, "r")
        skeleton = yaml.load(f, Loader=yaml.Loader)
        bvh_file = os.path.join(os.path.dirname(SKELETON_FILE), skeleton['BVH'])

        bvh = BVHReader(bvh_file)
        self.frame_time = bvh.frame_time
        self.offsets = bvh.get_offsets()
        self.parents = bvh.get_parents()
        self.joint_names = bvh.get_joint_names()
        self.rotation_order = bvh.get_rotation_order()

        self.left_fid, self.right_fid = skeleton['left_foot'], skeleton['right_foot']
        self.hips, self.shoulders = skeleton['hips'], skeleton['shoulders']
        self.head = skeleton['head']
        positions = self.offsets

        self.bone_length = calculate_bone_length_from_offsets(self.offsets, self.parents)
