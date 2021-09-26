import os

frame_time = "0.0333333"


def read_bvh_file(file):
    f = open(file, "r")
    frames = []
    names = []
    start_motion = False
    for line in f.readlines():
        str = line.strip('\n').strip()
        if start_motion:
            str_list = str.split(" ")

            frames.append(str_list)
        if str.startswith("JOINT") or str.startswith("ROOT"):
            str_list = str.split(" ")
            names.append(str_list[1])
        elif str.startswith("Frame Time"):
            start_motion = True
    return frames, names


def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh')]


def process_bvh(input_file, output_file, head):
    frames, joint_names = read_bvh_file(input_file)

    frame_num = len(frames)
    joint_num = int(len(joint_names))

    file = open(output_file, 'w')
    head += "Frames:	%d\n" % frame_num
    head += "Frame Time:	" + frame_time + "\n"
    file.write(head)

    for i, frame in enumerate(frames):
        for i in range(joint_num):
            s = ""
            if i == 0:
                s = frame[i * 6] + " " + frame[i * 6 + 1] + " " + frame[i * 6 + 2] + " " + \
                    frame[i * 6 + 3] + " " + frame[i * 6 + 4] + " " + frame[i * 6 + 5] + " "

            elif i not in delete_index:
                s = frame[i * 6 + 3] + " " + frame[i * 6 + 4] + " " + frame[i * 6 + 5] + " "
            file.write(s)
        file.write("\n")
    file.close()


# cyprus data process
input_path = "..\\data\\Preprocessing\\Cyprus BVH\\"
head_file = "..\\data\\Preprocessing\\head_file.bvh"
output_path = "..\\data\\Preprocessing\\Standrad BVH\\"

delete_index = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

head_in = open(head_file, "r")
head = ""
for line in head_in.readlines():
    head += line

bvh_files = get_bvh_files(input_path)
file_num = len(bvh_files)
for i, input_file in enumerate(bvh_files):
    print("Processing file %s (%i/%i)" % (input_file, i, file_num))
    strs = input_file.split('\\')
    name = strs[-1][0:-4]
    output_file = output_path + name + ".bvh"
    print("     out_file:", output_file)
    process_bvh(input_file, output_file, head)
