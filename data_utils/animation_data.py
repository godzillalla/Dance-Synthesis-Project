import numpy as np
from .skeleton import Skeleton
from .BVH_reader import BVHReader
import scipy.ndimage.filters as filters
from .quaternion_frame import convert_euler_frames_to_quaternion_frames
from .utils import quaternions_matrix, quaternion_between, from_quaternions, quaternions_cross_mul, \
    pivots_to_quaternions


class UnitData:
    def __init__(self, rot=None, quaternion=None, joint_pos=None, changed_quaternion=None, changed_pos=None,
                 root_pos=None, root_rot=None, contact=None, velocity=None, acceleration=None, vel_factor=None):
        self.rot = rot
        self.quaternion = quaternion
        self.joint_pos = joint_pos
        self.changed_quaternion = changed_quaternion
        self.changed_pos = changed_pos
        self.root_pos = root_pos
        self.contact = contact
        self.root_rot = root_rot
        self.velocity = velocity
        self.acceleration = acceleration
        self.vel_factor = vel_factor


def calculate_quaternions(rotations, rotation_order):
    quaternions = convert_euler_frames_to_quaternion_frames(rotations, rotation_order)
    return np.array(quaternions)


def get_positions(quaternions, skeleton):
    transforms = quaternions_matrix(quaternions)  # [..., J, 3, 3]
    global_positions = np.zeros(quaternions.shape[:-1] + (3,))  # [T, J, 3]
    for i, pi in enumerate(skeleton.parents):
        if pi == -1:
            continue
        global_positions[..., i, :] = np.matmul(transforms[..., pi, :, :],
                                                skeleton.offsets[i])
        global_positions[..., i, :] += global_positions[..., pi, :]
        transforms[..., i, :, :] = np.matmul(transforms[..., pi, :, :],
                                             transforms[..., i, :, :])
    return global_positions


def get_positions_2(quaternions, skeleton):
    positions = []
    for idx, frame in enumerate(quaternions):
        transforms = []
        for quaternion in frame:
            transforms.append(quaternions_matrix(quaternion))
        position = np.zeros(frame.shape[:-1] + (3,))  # J, 3
        for i, parent in enumerate(skeleton.parents):
            if parent == -1:
                continue
            position[i] = np.matmul(transforms[parent], skeleton.offsets[i])
            position[i] += position[parent]
            transforms[i] = np.matmul(transforms[parent], transforms[i])
        positions.append(position)
    return positions


def calculate_foot_contacts(positions, left_foot, right_foot):
    left_foot, right_foot = np.array(left_foot), np.array(right_foot)
    vel_factor = np.array([0.2, 0.2])
    feet_contact = []
    for index in [left_foot, right_foot]:
        foot_vel = (positions[1:, index] - positions[:-1, index]) ** 2  # T-1, 2, 3
        foot_vel = np.sum(foot_vel, axis=-1)  # T - 1, 2
        foot_contact = (foot_vel < vel_factor).astype(np.float)
        feet_contact.append(foot_contact)
    feet_contact = np.concatenate(feet_contact, axis=-1)
    feet_contact = np.concatenate((feet_contact[0:1].copy(), feet_contact), axis=0)
    return feet_contact  # [T, 4]


def across_from_pos(positions, hips, shoulders):
    across = positions[..., hips[0], :] - positions[..., hips[1], :] + \
             positions[..., shoulders[0], :] - positions[..., shoulders[1], :]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    return across


def y_rotation_from_positions(positions, hips, shoulders):
    """
    input: positions [T, J, 3]
    output: quaters: [T, 1, 4], quaternions that rotate the character around the y-axis to face [0, 0, 1]
            pivots: [T, 1] in [0, 2pi], the angle from [0, 0, 1] to the current facing direction
    """
    across = across_from_pos(positions, hips, shoulders)
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.tile(np.array([0, 0, 1]), forward.shape[:-1] + (1,))

    quaters = quaternion_between(forward, target)[..., np.newaxis, :]  # [T, 4] -> [T, 1, 4]
    pivots = from_quaternions(-quaters)  # from "target"[0, 0, 1] to current facing direction "forward"
    return quaters, pivots


def get_origin_rotations(changed_rotations, pivots):
    '''
    :param changed_rotations: [T, J, 4] the quaternions of character facing[0, 0, 1]
    :param pivots: [T, 1] in [0, 2pi], the angle from [0, 0, 1] to the current facing direction
    :return: origin rotations
    '''
    yaxis_rotations = np.array((pivots_to_quaternions(pivots)))
    rt_rotations = changed_rotations[:, :1]  # [T, 1, 4]
    rt_rotations = np.array(quaternions_cross_mul(yaxis_rotations, rt_rotations))
    rt_rotations /= np.sqrt((rt_rotations ** 2).sum(axis=-1))[..., np.newaxis]
    origin_rotations = np.concatenate((rt_rotations, changed_rotations[:, 1:]), axis=1)
    return origin_rotations  # [T, J, 4]


def face_z_directions(positions, quaternions, skeleton):
    quaters, pivots = y_rotation_from_positions(positions, skeleton.hips, skeleton.shoulders)

    changed_quaternions = quaternions.copy()
    changed_quaternions /= np.sqrt(np.sum(changed_quaternions ** 2, axis=-1))[..., np.newaxis]
    root_quaternions = changed_quaternions[:, 0:1, :].copy()  # [T, 1, 4]
    root_quaternions = quaternions_cross_mul(quaters, root_quaternions)  # facing [0, 0, 1]
    root_quaternions = np.array(root_quaternions).reshape((-1, 1, 4))  # [T, 1, 4]
    changed_quaternions[:, 0:1, :] = root_quaternions
    changed_quaternions.reshape(len(quaternions), -1)

    changed_positions = get_positions(changed_quaternions, skeleton)
    return changed_quaternions, changed_positions, pivots


def test_unit_data(unit):
    print("-------------------------------unit data-------------------------------")
    # rot, quaternion, joint_pos, changed_quaternion, changed_pos,
    # root_pos, root_rot, contact, velocity, acceleration
    print("rot.shape", unit.rot.shape)
    print("quaternion.shape", unit.quaternion.shape)
    print("joint_pos.shape", unit.joint_pos.shape)
    print("changed_pos.shape", unit.changed_pos.shape)
    print("changed_quaternion.shape", unit.changed_quaternion.shape)
    print("root_rot.shape", unit.root_pos.shape)
    print("root_rot.shape", unit.root_rot.shape)
    print("contact.shape", unit.contact.shape)
    print("velocity.shape", unit.velocity.shape)
    print("acceleration.shape", unit.acceleration.shape)
    print()


def get_velocity_acceleration(positions):
    temp_positions = np.concatenate([[positions[0]], positions], axis=0)  # [T + 1, J, 3]
    velocity = temp_positions[1:] - temp_positions[:-1]  # [T, J, 3]
    temp_velocity = np.concatenate([[velocity[0]], velocity], axis=0)  # [T + 1, J]
    acceleration = temp_velocity[1:] - temp_velocity[:-1]  # [T, J]
    return velocity, acceleration


def get_vel_factor(velocity):
    seq_len = velocity.shape[0]
    joint_num = velocity.shape[1]

    weight = [1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1]
    parts = [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0]
    weight_sum = []
    for part in range(5):
        p_sum = 0
        for j in range(joint_num):
            if parts[j] == part:
                p_sum += weight[j]
        weight_sum.append(p_sum)

    vel_factor = []
    for i in range(seq_len):
        part_factor = []
        for part in range(5):
            factor = 0
            for j in range(joint_num):
                if parts[j] == part:
                    factor += weight[j] / weight_sum[part] * \
                              pow(pow(velocity[i][j][0], 2) + pow(velocity[i][j][1], 2) + pow(velocity[i][j][2], 2),
                                  0.5)
            part_factor.append(factor)

        vel_factor.append(part_factor)
    vel_factor = np.array(vel_factor)
    return vel_factor


class AnimaitonData:

    def __init__(self, data=None, filename=None, downsample=1, skeleton=None):

        if skeleton is None:
            self.skeleton = Skeleton()
        else:
            self.skeleton = skeleton
        if filename is None and data is None:
            print("Error: no data", filename, data)
            exit(0)

        if filename is None:
            self.unit = data
        else:
            strs = filename.split("\\")
            print("read animation data of", strs[-1])
            bvh = BVHReader(filename)
            root_pos, rotation = bvh.get_data()
            quaternion = calculate_quaternions(rotation, self.skeleton.rotation_order)
            joint_pos = get_positions(quaternion, self.skeleton)
            changed_quaternion, changed_pos, root_rot = face_z_directions(joint_pos, quaternion,
                                                                          self.skeleton)
            contact = calculate_foot_contacts(changed_pos, self.skeleton.left_fid, self.skeleton.right_fid)

            pos = np.zeros(changed_pos.shape)
            pos[:, :-1, :] = changed_pos[:, 1:, :]
            pos[:, -1, :] = root_pos
            velocity, acceleration = get_velocity_acceleration(pos)
            vel_factor = get_vel_factor(velocity)

            self.unit = UnitData(rot=rotation, quaternion=quaternion, joint_pos=joint_pos[:, 1:, :],
                                 changed_quaternion=changed_quaternion, changed_pos=changed_pos[:, 1:, :],
                                 root_pos=root_pos, root_rot=root_rot, contact=contact,
                                 velocity=velocity, acceleration=acceleration, vel_factor=vel_factor)

        self.down_sample_unit = self.down_sample_data(downsample)

    def get_down_sample_data(self):
        return self.down_sample_unit

    def get_full_data(self):
        return self.unit

    def down_sample_data(self, downsample):
        if downsample == 1:
            self.down_sample_unit = self.unit
        else:
            changed_pos = self.unit.changed_pos[::downsample]
            velocity, acceleration = get_velocity_acceleration(changed_pos)
            vel_factor = get_vel_factor(velocity)
            self.down_sample_unit = UnitData(rot=self.unit.rot[:: downsample],
                                             quaternion=self.unit.quaternion[::downsample],
                                             joint_pos=self.unit.joint_pos[::downsample],
                                             changed_quaternion=self.unit.changed_quaternion[::downsample],
                                             changed_pos=self.unit.changed_pos[::downsample],
                                             root_pos=self.unit.root_pos[::downsample],
                                             root_rot=self.unit.root_rot[::downsample],
                                             contact=self.unit.contact[::downsample],
                                             velocity=velocity,
                                             acceleration=acceleration,
                                             vel_factor=vel_factor)

        return self.down_sample_unit
