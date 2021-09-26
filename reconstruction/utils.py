import numpy as np
import math


def QuaToEuler(rot_angle, rot_axis):  # 'xyz'
    q0 = math.cos(rot_angle / 2)
    q1 = rot_axis[0] * math.sin(rot_angle / 2)
    q2 = rot_axis[1] * math.sin(rot_angle / 2)
    q3 = rot_axis[2] * math.sin(rot_angle / 2)
    euler = np.zeros(3)
    euler[2] = math.atan(2 * (q1 * q2 - q0 * q3) / (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3))
    euler[1] = math.asin(-2 * (q0 * q2 + q1 * q3))
    euler[0] = math.atan(2 * (q2 * q3 - q0 * q1) / (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3))
    # print ('euler:')
    # print (euler)
    return euler


def rotateMatrix(rot_axis, rot_angle):
    """
    Rotate a vector along the rotation axis counterclockwise.
    :param vector: the vector needs to rotate
    :param rot_axis: the rotation axis needs to be a unit vector
    :param rot_angle: rotation angle
    :return: the vector after rotating
    """
    rot_matrix = np.zeros((3, 3))
    cos_theta = math.cos(rot_angle)
    sin_theta = math.sin(rot_angle)

    """Rotation matrix, reference: UCSD, chapter 04"""
    rot_matrix[0, 0] = rot_axis[0] * rot_axis[0] + cos_theta * (1 - rot_axis[0] * rot_axis[0])
    rot_matrix[0, 1] = rot_axis[0] * rot_axis[1] * (1 - cos_theta) + rot_axis[2] * sin_theta
    rot_matrix[0, 2] = rot_axis[0] * rot_axis[2] * (1 - cos_theta) - rot_axis[1] * sin_theta
    rot_matrix[1, 0] = rot_axis[0] * rot_axis[1] * (1 - cos_theta) - rot_axis[2] * sin_theta
    rot_matrix[1, 1] = rot_axis[1] * rot_axis[1] + cos_theta * (1 - rot_axis[1] * rot_axis[1])
    rot_matrix[1, 2] = rot_axis[1] * rot_axis[2] * (1 - cos_theta) + rot_axis[0] * sin_theta
    rot_matrix[2, 0] = rot_axis[0] * rot_axis[2] * (1 - cos_theta) + rot_axis[1] * sin_theta
    rot_matrix[2, 1] = rot_axis[1] * rot_axis[2] * (1 - cos_theta) - rot_axis[0] * sin_theta
    rot_matrix[2, 2] = rot_axis[2] * rot_axis[2] + cos_theta * (1 - rot_axis[2] * rot_axis[2])

    return rot_matrix


def rotation_order_to_string(rotation_order):
    r_order_string = "r"
    for c in rotation_order:
        if c == "Xrotation":
            r_order_string += "x"
        elif c == "Yrotation":
            r_order_string += "y"
        elif c == "Zrotation":
            r_order_string += "z"
    return r_order_string


def calculate_diatance_3d(a, b):
    delta_x = a[0] - b[0]
    delta_y = a[1] - b[1]
    delta_z = a[2] - a[2]
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
