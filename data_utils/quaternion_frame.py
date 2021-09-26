import numpy as np
from .constants import DEFAULT_ROTATION_ORDER
from transformations import euler_matrix, quaternion_from_matrix, quaternion_matrix, euler_from_matrix


def rotation_order_to_string(rotation_order, init_str="r"):
    r_order_string = init_str
    for c in rotation_order:
        if c == "Xrotation":
            r_order_string += "x"
        elif c == "Yrotation":
            r_order_string += "y"
        elif c == "Zrotation":
            r_order_string += "z"
    return r_order_string


def convert_quat_frame_value_to_array(quat_frame_values):
    n_channels = len(quat_frame_values)
    quat_frame_value_array = []
    for item in quat_frame_values:
        if not isinstance(item, list):
            item = list(item)
        quat_frame_value_array += item
    return quat_frame_value_array


def check_quat(test_quat, ref_quat):
    """check locomotion_synthesis_test quat needs to be filpped or not
    """
    test_quat = np.asarray(test_quat)
    ref_quat = np.asarray(ref_quat)
    dot = np.dot(test_quat, ref_quat)
    if dot < 0:
        test_quat = - test_quat
    return test_quat.tolist()


def euler_to_quaternion(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER, filter_values=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]
    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in degress
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    """
    assert len(euler_angles) == 3, ('The length of euler angles should be 3!')
    euler_angles = np.deg2rad(euler_angles)
    rotmat = euler_matrix(*euler_angles, rotation_order_to_string(rotation_order))
    # convert rotation matrix R into quaternion vector (qw, qx, qy, qz)
    quat = quaternion_from_matrix(rotmat)
    # filter the quaternion see
    # http://physicsforgames.blogspot.de/2010/02/quaternions.html
    if filter_values:
        dot = np.sum(quat)
        if dot < 0:
            quat = -quat
    return [quat[0], quat[1], quat[2], quat[3]]


def quaternion_to_euler(quat, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    Parameters
    ----------
    * q: list of floats
    \tQuaternion vector with form: [qw, qx, qy, qz]

    Return
    ------
    * euler_angles: list
    \tEuler angles in degree with specified order
    :param rotation_order:
    """
    quat = np.asarray(quat)
    quat = quat / np.linalg.norm(quat)
    rotmat_quat = quaternion_matrix(quat)
    rotation_order_str = rotation_order_to_string(rotation_order)
    euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, rotation_order_str))
    return euler_angles


def convert_euler_to_quaternion_frame(e_frame, order=DEFAULT_ROTATION_ORDER, filter_values=True):
    """Convert a BVH frame into an ordered dict of quaternions for each skeleton node
    Parameters
    ----------
    * frame_vector: np.ndarray
    \t animation keyframe frame represented by Euler angles
    *
    * filter_values: Bool
    \t enforce a unique rotation representation

    Returns:
    ----------
    * quat_frame: list that contains a quaternion for each joint

    """
    quat_frame = []
    for i in range(len(e_frame)):
        angle = e_frame[i]
        quat_frame.append(euler_to_quaternion(angle, order, filter_values))
    return quat_frame


def convert_euler_frames_to_quaternion_frames(euler_frames, order=DEFAULT_ROTATION_ORDER, filter_values=True):
    """
    :param euler_frames: a list of euler frames
    :return: a list of quaternion frames
    """
    quat_frames = []
    prev_frame_list = None
    for e_frame in euler_frames:
        quat_frame_list = convert_euler_to_quaternion_frame(e_frame, order, filter_values)
        if prev_frame_list is not None and filter_values:
            for i, quat in enumerate(quat_frame_list):
                q = check_quat(quat, prev_frame_list[i])
                quat_frame_list[i] = q
        prev_frame_list = quat_frame_list
        quat_frames.append(quat_frame_list)
    return quat_frames


def convert_quaternion_frames_to_euler_frames(quaternion_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------

    :param quaternion_frames:
     * quaternion_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------

    * euler_frames: numpy array
    \tEuler frames
    """

    def gen_4_tuples(it):
        """Generator of n-tuples from iterable"""

        return list(zip(it[0::4], it[1::4], it[2::4], it[3::4]))

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""
        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler = quaternion_to_euler(quaternion)
            euler_frame.append(euler[0])
            euler_frame.append(euler[1])
            euler_frame.append(euler[2])
        return euler_frame

    euler_frames = list(map(get_euler_frame, quaternion_frames))
    return np.array(euler_frames)


def convert_quaternion_to_euler(quaternion_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------

     * quaternion_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------

    * euler_frames: numpy array
    \tEuler frames
    """

    def gen_4_tuples(it):
        """Generator of n-tuples from iterable"""

        return list(zip(it[0::4], it[1::4], it[2::4], it[3::4]))

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""

        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler_frame += quaternion_to_euler(quaternion)

        return euler_frame

    euler_frames = list(map(get_euler_frame, quaternion_frames))

    return np.array(euler_frames)
