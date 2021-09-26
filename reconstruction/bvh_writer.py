# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

from data_utils.constants import DEFAULT_ROTATION_ORDER


class BVHWriter(object):
    def __init__(self, rotation, file_name, joint_name, children_list, initial_offset, frame_per_second):
        self.rotation = rotation
        self.file_name = file_name
        self.joints = joint_name
        self.offset = initial_offset
        self.children = children_list
        self.fps = frame_per_second

    def write_to_bvh(self):
        out_data = self.get_bvh_info() + self.get_rotation_data()
        with open(self.file_name, 'w') as file_object:
            file_object.write(out_data)

    def get_bvh_info(self):
        line_break = "\n"

        bvh_info = "HIERARCHY" + line_break
        bvh_info += self.get_info_for_one_joint(0, 0)
        bvh_info += "MOTION" + line_break
        bvh_info += "Frames: " + str(len(self.rotation)) + line_break
        bvh_info += "Frame Time: " + str(1 / self.fps) + line_break
        return bvh_info

    def get_rotation_data(self):
        rot_data = ""
        line_break = "\n"
        for rotation in self.rotation:
            for rot in rotation:
                rot_data += str(rot[0]) + " " + str(rot[1]) + " " + str(rot[2]) + " "
            rot_data += line_break
        return rot_data

    def get_info_for_one_joint(self, indent_num, index):
        indent = ""
        for i in range(indent_num):
            indent += "    "
        line_break = "\n"
        info = ""
        joint = self.joints[index]
        if index == 0:
            info += "ROOT " + joint + line_break
            info += "{" + line_break
            info += "    " + "OFFSET " + str(self.offset[index][0]) + \
                    " " + str(self.offset[index][1]) + " " + str(self.offset[index][2]) + line_break
            info += "    " + "CHANNELS 6 Xposition Yposition Zposition " + \
                    DEFAULT_ROTATION_ORDER[0] + " " + DEFAULT_ROTATION_ORDER[1] + " " + DEFAULT_ROTATION_ORDER[
                        2] + line_break
            for child in self.children[index]:
                info += self.get_info_for_one_joint(indent_num + 1, child)
        else:
            info += indent + "JOINT " + joint + line_break
            info += indent + "{" + line_break
            info += "    " + indent + "OFFSET " + str(self.offset[index][0]) + \
                    " " + str(self.offset[index][1]) + " " + str(self.offset[index][2]) + line_break
            info += "    " + indent + "CHANNELS 3 " + DEFAULT_ROTATION_ORDER[0] + \
                    " " + DEFAULT_ROTATION_ORDER[1] + " " + DEFAULT_ROTATION_ORDER[2] + line_break
            if len(self.children[index]) == 0:
                info += "    " + indent + "End Site" + line_break
                info += "    " + indent + "{" + line_break
                info += "        " + indent + "OFFSET 0 0 0" + line_break
                info += "    " + indent + "}" + line_break
            else:
                for child in self.children[index]:
                    info += self.get_info_for_one_joint(indent_num + 1, child)
        info += indent + "}" + line_break
        return info
