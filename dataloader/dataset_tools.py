import os
from os.path import join, exists
import sys
import argparse
import numpy as np
import glob


def compute_ck_action_unit(image_root_dir, output_dir, exec):
    image_files = []
    for dirpath, dirname, filenames in os.walk(image_root_dir):
        if len(filenames) > 0:
            for f in filenames:
                image_files += [join(dirpath, f)]

    for f in image_files:
        os.system(exec + ' -aus -au_static -f ' + f + ' -out_dir ' + output_dir)


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

    args = parser.parse_args(argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))


if __name__ == '__main__':
    execute_cmdline(sys.argv)
