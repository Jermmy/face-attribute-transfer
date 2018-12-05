import os
from os.path import join, exists
import sys
import argparse
import random
import cv2
import numpy as np


def compute_ck_action_unit(image_root_dir, output_dir, exec):
    image_files = []
    for dirpath, dirname, filenames in os.walk(image_root_dir):
        if len(filenames) > 0:
            for f in filenames:
                if f.endswith('png'):
                    image_files += [join(dirpath, f)]

    if not exists(output_dir):
        os.makedirs(output_dir)

    for f in image_files:
        os.system(exec + ' -aus -au_static -f ' + f + ' -out_dir ' + output_dir + ' -q')


# def compute_ck_face_landmark(image_root_dir, output_dir, exec):
#     image_files = []
#     for dirpath, dirname, filenames in os.walk(image_root_dir):
#         if len(filenames) > 0:
#             for f in filenames:
#                 if f.endswith('png'):
#                     image_files += [join(dirpath, f)]
#
#     if not exists(output_dir):
#         os.makedirs(output_dir)
#
#     for f in image_files:
#         os.system(exec + ' -2Dfp -f ' + f + ' -out_dir ' + output_dir + ' -q')


def clip_ck_face(image_root_dir, landmark_root_dir, output_dir):
    image_files = []
    for dirpath, dirname, filenames in os.walk(image_root_dir):
        if len(filenames) > 0:
            for f in filenames:
                if f.endswith('png'):
                    image_files += [join(dirpath, f)]

    if not exists(output_dir):
        os.makedirs(output_dir)

    image_root_name = image_root_dir.split('/')[-1]
    landmark_root_name = landmark_root_dir.split('/')[-1]
    for image_file in image_files:
        print('Process %s' % image_file)
        image = cv2.imread(image_file)

        landmark_file = image_file.replace(image_root_name, landmark_root_name).split('.png')[0]\
                        + '_landmarks.txt'
        landmarks = []
        with open(landmark_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split('   ')
                landmarks += [(int(float(line[0])), int(float(line[1])))]
        landmarks = np.array(landmarks)
        tl = np.min(landmarks, axis=0)   # (x, y)
        br = np.max(landmarks, axis=0)
        height = br[1] - tl[1]
        width = br[0] - tl[0]
        image = image[max(tl[1] - height // 4, 0): br[1],
                max(tl[0] - width // 10, 0): min(br[0] + width // 10, image.shape[1])]
        cv2.imwrite(join(output_dir, image_file.split('/')[-1]), image)


def split_dataset(image_dir, train_file, test_file):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('png')]
    random.shuffle(image_files)
    train_image_files = image_files[0: int(len(image_files) * 0.8)]
    test_image_files = image_files[int(len(image_files) * 0.8):]

    with open(train_file, 'w') as file:
        for f in train_image_files:
            au_file = f.replace('png', 'csv')
            file.write(f + ',' + au_file + '\n')

    with open(test_file, 'w') as file:
        for f in test_image_files:
            au_file = f.replace('png', 'csv')
            file.write(f + ',' + au_file + '\n')


def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog=prog,
        description='Tool to create Action Units.'
    )

    subparsers = parser.add_subparsers(dest='command')

    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('compute_ck_action_unit', 'Compute CK+ action units.',
                    'compute_ck_action_unit cohn-kanade-images action-units OpenFace/build/bin/FeatureExtraction')
    p.add_argument('image_root_dir', help='Root directory to read CK+ image files.')
    p.add_argument('output_dir', help='Directory to write Action Unit files.')
    p.add_argument('exec', help='Executable file.')

    p = add_command('clip_ck_face', 'Clip CK+ image face.',
                    'clip_ck_face cohn-kanade-images landmarks clip-faces')
    p.add_argument('image_root_dir', help='Root directory to read CK+ image files.')
    p.add_argument('landmark_root_dir', help='Root directory to read landmark files.')
    p.add_argument('output_dir', help='Directory to write face images.')

    # p = add_command('compute_ck_face_landmark', 'Compute CK+ landmarks.',
    #                 'compute_ck_face_landmark cohn-kanade-images landmarks OpenFace/build/bin/FaceLandmarkImg')
    # p.add_argument('image_root_dir', help='Root directory to read CK+ image files.')
    # p.add_argument('output_dir', help='Directory to write landmark files.')
    # p.add_argument('exec', help='Executable file.')

    p = add_command('split_dataset', 'Split CK+ dataset into train and test parts.',
                    'split_dataset clip_faces train_filelist test_filelist.')
    p.add_argument('image_dir', help='Root directory to read clip face image files.')
    p.add_argument('train_file', help='Train filelist to write.')
    p.add_argument('test_file', help='Test filelist to write.')

    args = parser.parse_args(argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))


if __name__ == '__main__':
    execute_cmdline(sys.argv)
