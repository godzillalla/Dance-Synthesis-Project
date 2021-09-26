import numpy as np
import math
from transformations import quaternion_slerp

BLEND_DIRECTION_FORWARD = 0
BLEND_DIRECTION_BACKWARD = 1


def quaternions_matrix(quaternions):
    qw = quaternions[..., 0]
    qx = quaternions[..., 1]
    qy = quaternions[..., 2]
    qz = quaternions[..., 3]

    xx2 = qx * qx * 2
    yy2 = qy * qy * 2
    wx2 = qw * qx * 2
    xy2 = qx * qy * 2
    yz2 = qy * qz * 2
    wy2 = qw * qy * 2
    xz2 = qx * qz * 2
    zz2 = qz * qz * 2
    wz2 = qw * qz * 2

    m = np.empty(quaternions.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1.0 - yy2 - zz2
    m[..., 0, 1] = xy2 - wz2
    m[..., 0, 2] = xz2 + wy2
    m[..., 1, 0] = xy2 + wz2
    m[..., 1, 1] = 1 - xx2 - zz2
    m[..., 1, 2] = yz2 - wx2
    m[..., 2, 0] = xz2 - wy2
    m[..., 2, 1] = yz2 + wx2
    m[..., 2, 2] = 1 - xx2 - yy2
    return m


def quaternions_cross_mul(quat1, quat2):
    w1 = quat1[..., 0]
    x1 = quat1[..., 1]
    y1 = quat1[..., 2]
    z1 = quat1[..., 3]

    w2 = quat2[..., 0]
    x2 = quat2[..., 1]
    y2 = quat2[..., 2]
    z2 = quat2[..., 3]
    res = np.empty(quat1.shape)
    res[..., 0] = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    res[..., 1] = w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
    res[..., 2] = w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1
    res[..., 3] = w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1
    return res


def normalize_quaternion(q):
    length = np.sum(q ** 2, axis=-1) ** 0.5
    return q / length[..., np.newaxis]


def quaternion_between(q1, q2):
    a = np.cross(q1, q2)
    w = np.sqrt((q1 ** 2).sum(axis=-1) * (q2 ** 2).sum(axis=-1)) + (q1 * q2).sum(axis=-1)
    result = np.concatenate([w[..., np.newaxis], a], axis=-1)
    return normalize_quaternion(result)


def from_directions(ds, plane='xz'):
    ys = ds[..., 'xyz'.index(plane[0])]
    xs = ds[..., 'xyz'.index(plane[1])]
    res = np.array(np.arctan2(ys, xs))
    return res


def from_quaternions(qs, forward='z', plane='xz'):
    ds = np.zeros(qs.shape[:-1] + (3,))
    ds[..., 'xyz'.index(forward)] = 1.0

    vs = np.concatenate([np.zeros(ds.shape[:-1] + (1,)), ds], axis=-1)
    temp = quaternions_cross_mul(quaternions_cross_mul(vs, qs), qs)
    mask = np.ones(temp.shape, dtype=bool)
    mask[..., 0] = False
    res = temp[mask].reshape(ds.shape)
    return from_directions(res, plane=plane)


def quaternion_from_angle_axis(angles, axis):
    axis = axis / (np.sqrt(np.sum(axis ** 2, axis=-1)) + 1e-10)[..., np.newaxis]
    sines = np.sin(angles / 2.0)[..., np.newaxis]
    cosines = np.cos(angles / 2.0)[..., np.newaxis]
    return np.concatenate([cosines, axis * sines], axis=-1)


def pivots_to_quaternions(pivots, plane='xz'):
    fa = tuple(map(lambda x: slice(None), pivots.shape))
    axises = np.ones(pivots.shape + (3,))
    axises[fa + ("xyz".index(plane[0]),)] = 0.0
    axises[fa + ("xyz".index(plane[1]),)] = 0.0
    return quaternion_from_angle_axis(pivots, axises)


def get_distance_3d_point(a, b):
    delta_x = a[0] - b[0]
    delta_y = a[1] - b[1]
    delta_z = a[2] - b[2]
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


def blend_quaternion(a, b, w):
    return quaternion_slerp(a, b, w, spin=0, shortestpath=True)


def smooth_joints_around_transition_using_slerp(quat_frames, joint_param_indices, discontinuity, window):
    h_window = int(window / 2)
    start_frame = max(discontinuity - h_window, 0)
    end_frame = min(discontinuity + h_window, len(quat_frames) - 1)
    start_window = discontinuity - start_frame
    end_window = end_frame - discontinuity
    if start_window > 0:
        create_transition_for_joints_using_slerp(quat_frames, joint_param_indices, start_frame, discontinuity,
                                                 start_window, BLEND_DIRECTION_FORWARD)
    if end_window > 0:
        create_transition_for_joints_using_slerp(quat_frames, joint_param_indices, discontinuity, end_frame, end_window,
                                                 BLEND_DIRECTION_BACKWARD)


def create_transition_for_joints_using_slerp(quat_frames, joint_param_indices, start_frame, end_frame, steps,
                                             direction=BLEND_DIRECTION_FORWARD):
    new_quats = create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_param_indices)
    for i in range(steps):
        if direction == BLEND_DIRECTION_FORWARD:
            t = float(i) / steps
        else:
            t = 1.0 - (i / steps)
        old_quat = quat_frames[start_frame + i, joint_param_indices]
        blended_quat = blend_quaternion(old_quat, new_quats[i], t)
        quat_frames[start_frame + i, joint_param_indices] = blended_quat
    return quat_frames


def smooth_root_translation_around_transition(frames, d, window):
    hwindow = int(window / 2.0)
    # root_pos1 = frames[d - 1, :3]
    # root_pos2 = frames[d, :3]
    # root_pos = (root_pos1 + root_pos2) / 2
    root_pos = frames[d, :3]
    start_idx = d - hwindow
    end_idx = d + hwindow
    start = frames[start_idx, :3]
    end = root_pos
    for i in range(hwindow):
        t = float(i) / hwindow
        frames[start_idx + i, :3] = start * (1 - t) + end * t
    start = root_pos
    if end_idx < len(frames):
        end = frames[end_idx, :3]
        for i in range(hwindow):
            t = float(i) / hwindow
            frames[d + i, :3] = start * (1 - t) + end * t


def create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_parameter_indices):
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    frames = []
    for i in range(steps):
        t = float(i) / steps
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        frames.append(slerp_q)
    return frames
