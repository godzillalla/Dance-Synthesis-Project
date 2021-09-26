

import os
from collections import OrderedDict
import numpy as np
from transformations import euler_from_matrix, euler_matrix
from .quaternion_frame import rotation_order_to_string


class BVHReader(object):

    def __init__(self, filename=""):
        self.node_names = OrderedDict()
        self.node_channels = []
        self.rotation_order = []
        self.offsets = []
        self.joint_names = []
        self.parents = []
        self.frame_time = None
        self.frames = None
        self.root = ""
        if filename != "":
            infile = open(filename, "r")
            lines = infile.readlines()
            _lines = []
            for l in lines:
                if l.strip() != "":
                    _lines.append(l)
            self.process_lines(_lines)
            infile.close()
        self.filename = os.path.split(filename)[-1]


    @classmethod
    def init_from_string(cls, skeleton_string):
        bvh_reader = cls(filename="")
        lines = skeleton_string.split("\n")
        bvh_reader.process_lines(lines)
        return bvh_reader

    def get_data(self):
        positions = []
        rotations = []
        names = self.get_joint_names()

        for frame in self.frames:
            positions.append([frame[0], frame[1], frame[2]])
            temp_rotation = []
            for i in range(len(names)):
                temp_rotation.append([frame[i * 3 + 3], frame[i * 3 + 4], frame[i * 3 + 5]])
            rotations.append(temp_rotation)

        return np.array(positions), np.array(rotations)


    def get_rotation_order(self):
        if len(self.rotation_order) == 0:
            for ch in self.node_names[self.root]["channels"]:
                if ch.lower().endswith("rotation"):
                    self.rotation_order.append(ch)
        return self.rotation_order

    def get_offsets(self):
        if len(self.offsets) == 0:
            for key in self.node_names.keys():
                if not key.endswith("_EndSite"):
                    self.offsets.append(self.node_names[key]["offset"])
        return np.array(self.offsets)

    def get_joint_names(self):
        if len(self.joint_names) == 0:
            for key in self.node_names.keys():
                if not key.endswith("_EndSite"):
                    self.joint_names.append(key)
        return self.joint_names

    def get_parents(self):
        joint_names = self.get_joint_names()
        if len(self.parents) == 0:
            parent_dict = {}
            for name in self.node_names.keys():
                if "children" in self.node_names[name]:
                    for child in self.node_names[name]["children"]:
                        parent_dict[child] = name
            for name in joint_names:
                if name in parent_dict.keys():
                    parent = parent_dict[name]
                    self.parents.append(joint_names.index(parent))
                else:
                    self.parents.append(-1)
        return self.parents

    def _read_skeleton(self, lines, line_index=0, n_lines=-1):
        """Reads the skeleton part of a BVH file"""
        line_index = line_index
        parents = []
        level = 0
        name = None
        if n_lines == -1:
            n_lines = len(lines)

        while line_index < n_lines:
            if lines[line_index].startswith("MOTION"):
                break

            else:
                if "{" in lines[line_index]:
                    parents.append(name)
                    level += 1

                if "}" in lines[line_index]:
                    level -= 1
                    parents.pop(-1)
                    if level == 0:
                        break

                line_split = lines[line_index].strip().split()

                if line_split:

                    if line_split[0] == "ROOT":
                        name = line_split[1]
                        self.root = name
                        self.node_names[name] = {
                            "children": [], "level": level, "channels": [], "channel_indices": []}

                    elif line_split[0] == "JOINT":
                        name = line_split[1]
                        self.node_names[name] = {
                            "children": [], "level": level, "channels": [], "channel_indices": []}
                        self.node_names[parents[-1]]["children"].append(name)

                    elif line_split[0] == "CHANNELS":
                        for channel in line_split[2:]:
                            self.node_channels.append((name, channel))
                            self.node_names[name]["channels"].append(channel)
                            self.node_names[name]["channel_indices"].append(len(self.node_channels) - 1)

                    elif line_split == ["End", "Site"]:
                        name += "_" + "".join(line_split)
                        self.node_names[name] = {"level": level}
                        # also the end sites need to be adde as children
                        self.node_names[parents[-1]]["children"].append(name)

                    elif line_split[0] == "OFFSET" and name in list(self.node_names.keys()):
                        self.node_names[name]["offset"] = list(map(float, line_split[1:]))
                line_index += 1
        return line_index



    def _read_frametime(self, lines, line_index):
        """Reads the frametime part of a BVH file"""

        if lines[line_index].startswith("Frame Time:"):
            self.frame_time = float(lines[line_index].split(":")[-1].strip())
        else:
            self.frame_time = -1

    def _read_frames(self, lines, line_index, n_lines=-1):
        """Reads the frames part of a BVH file"""
        line_index = line_index
        if n_lines == -1:
            n_lines = len(lines)
        frames = []
        while line_index < n_lines:
            line_split = lines[line_index].strip().split()
            frames.append(np.array(list(map(float, line_split))))
            line_index += 1

        self.frames = np.array(frames)
        return line_index

    def process_lines(self, lines):
        """Reads BVH file infile

        Parameters
        ----------
         * infile: Filelike object, optional
        \tBVH file

        """
        line_index = 0
        n_lines = len(lines)
        while line_index < n_lines:
            if lines[line_index].startswith("HIERARCHY"):
                line_index = self._read_skeleton(lines, line_index, n_lines)
            if lines[line_index].startswith("MOTION"):
                self._read_frametime(lines, line_index+2)
                line_index = self._read_frames(lines, line_index+3, n_lines)
            else:
                line_index += 1

    def get_channel_indices(self, node_channels):
        """Returns indices for specified channels

        Parameters
        ----------
         * node_channels: 2-tuples of strings
        \tEach tuple contains joint name and channel name
        \te.g. ("hip", "Xposition")

        """
        return [self.node_channels.index(nc) for nc in node_channels]

    def get_node_channels(self, node_name):
        channels = None
        if node_name in self.node_names and "channels" in self.node_names[node_name]:
            channels = self.node_names[node_name]["channels"]
        return channels

    def get_node_angles(self, node_name, frame):
        """Returns the rotation for one node at one frame of an animation
        Parameters
        ----------
        * node_name: String
        \tName of node
        * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame: np.ndarray
        \t animation keyframe frame

        """
        channels = self.node_names[node_name]["channels"]
        euler_angles = []
        rotation_order = []
        for ch in channels:
            if ch.lower().endswith("rotation"):
                idx = self.node_channels.index((node_name, ch))
                rotation_order.append(ch)
                euler_angles.append(frame[idx])
        return euler_angles, rotation_order

    def get_angles(self, node_channels):
        """Returns numpy array of angles in all frames for specified channels

        Parameters
        ----------
         * node_channels: 2-tuples of strings
        \tEach tuple contains joint name and channel name
        \te.g. ("hip", "Xposition")

        """
        indices = self.get_channel_indices(node_channels)
        return self.frames[:, indices]

    def convert_rotation_order(self, rotation_order):
        self.convert_skeleton_rotation_order(rotation_order)
        self.convert_motion_rotation_order(rotation_order_to_string(rotation_order))

    def convert_skeleton_rotation_order(self, rotation_order):
        # update channel indices
        rotation_list = self.node_names[self.root]["channels"][3:]
        new_indices = sorted(range(len(rotation_list)), key=lambda k : rotation_list[k])
        for node_name, node in self.node_names.items():
            if 'End' not in node_name:
                if len(node['channels']) == 6:
                    rotation_list = node['channels'][3:]
                    node['channels'][3:] = rotation_order
                    node['rotation_order'] = rotation_order_to_string(rotation_list)
                else:
                    rotation_list = node['channels']
                    node['channels'] = rotation_order
                    node['rotation_order'] = rotation_order_to_string(rotation_list)


    def convert_motion_rotation_order(self, rotation_order_str):
        new_frames = np.zeros(self.frames.shape)
        for i in range(len(new_frames)):
            for node_name, node in self.node_names.items():
                if 'End' not in node_name:
                    if len(node['channels']) == 6:
                        rot_mat = euler_matrix(*np.deg2rad(self.frames[i, node['channel_indices'][3:]]),
                                               axes=node['rotation_order'])
                        new_frames[i, node['channel_indices'][:3]] = self.frames[i, node['channel_indices'][:3]]
                        new_frames[i, node['channel_indices'][3:]] = np.rad2deg(euler_from_matrix(rot_mat, rotation_order_str))
                    else:
                        rot_mat = euler_matrix(*np.deg2rad(self.frames[i, node['channel_indices']]),
                                               axes=node['rotation_order'])
                        new_frames[i, node['channel_indices']] = np.rad2deg(euler_from_matrix(rot_mat, rotation_order_str))
        self.frames = new_frames

    def scale(self, scale):
        for node in self.node_names:
            self.node_names[node]["offset"] = [scale * o for o in self.node_names[node]["offset"]]
            if "channels" not in self.node_names[node]:
                continue
            ch = [(node, c) for c in self.node_names[node]["channels"] if "position" in c]
            if len(ch) > 0:
                ch_indices = self.get_channel_indices(ch)
                scaled_params = [scale * o for o in self.frames[:, ch_indices]]
                self.frames[:, ch_indices] = scaled_params
