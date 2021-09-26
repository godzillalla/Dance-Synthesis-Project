
import os
BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ROTATION_ORDER = ['Zrotation', 'Xrotation', 'Yrotation']
DATA_DIRECTORY = BASEPATH + "/data/"
GLOBAL_INFO_DIRECTORY = BASEPATH + "/global_info/"
TEST_OUT_DIRECTORY = BASEPATH + "/test_out/"
SKELETON_FILE = BASEPATH + "/global_info/skeleton_cyprus.yml"
SAVE_YAML_FILE = "bone_info.yml"
