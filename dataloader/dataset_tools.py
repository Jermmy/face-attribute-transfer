import os
from os.path import join, exists
import sys
import argparse
import face_recognition
import cv2


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


def clip_ck_face(image_root_dir, output_dir):
    image_files = []
    for dirpath, dirname, filenames in os.walk(image_root_dir):
        if len(filenames) > 0:
            for f in filenames:
                if f.endswith('png'):
                    image_files += [join(dirpath, f)]

    if not exists(output_dir):
        os.makedirs(output_dir)

    for f in image_files:
        print('Process %s' % f)
        image = face_recognition.load_image_file(f)
        l = face_recognition.face_locations(image, model='cnn')[0]
        height = l[2] - l[0]
        face = image[max(l[0] - height // 4, 0): l[2], l[3]: l[1]]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        f = f.split('/')[-1]
        cv2.imwrite(join(output_dir, f), face)


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
                    'clip_ck_face cohn-kanade-images clip-face')
    p.add_argument('image_root_dir', help='Root directory to read CK+ image files.')
    p.add_argument('output_dir', help='Directory to write face images.')

    args = parser.parse_args(argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))


if __name__ == '__main__':
    execute_cmdline(sys.argv)
