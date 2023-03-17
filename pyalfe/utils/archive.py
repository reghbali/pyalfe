import glob
import os
import shutil
import string
import subprocess
import tarfile
import random


def generate_random_str(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def copy_matching_paths(src_pattern, dst, root_dir=''):
    if type(src_pattern) is str:
        matching_paths = glob.glob(os.path.join(root_dir, src_pattern))
    elif hasattr(src_pattern, '__iter__'):
        matching_paths = []
        for pattern in src_pattern:
            matching_paths += glob.glob(os.path.join(root_dir, pattern))
    else:
        raise ValueError('src_pattern should be either str or an Iterable.')

    if len(matching_paths) == 0:
        raise ValueError(f'No path matching {src_pattern}')
    for src in matching_paths:
        shutil.copy(src, dst)


def extract_binary_from_archive(archive_path, dst, binary_name):
    archive_name, archive_type = os.path.basename(archive_path).split('.')[0:2]
    extract_path = os.path.join('/tmp', archive_name + generate_random_str(9))
    if archive_type == 'dmg':
        extract_file_from_dmg(
            archive_path,
            extract_path,
            dst,
            [
                os.path.join('*.app', 'Contents', 'bin', binary_name),
                os.path.join('bin', binary_name),
            ],
        )
    elif archive_type == 'tar':
        extract_file_from_tar(
            archive_path, extract_path, dst, os.path.join('*', 'bin', binary_name)
        )
    else:
        raise ValueError(
            f'cannot extract from {archive_path}'
            f' unsupported extension {archive_type}'
        )


def extract_file_from_archive(archive_path, dst, relative_file_path):
    archive_name, archive_type = os.path.basename(archive_path).split('.')[0:2]
    extract_path = os.path.join('/tmp', archive_name + generate_random_str(9))
    if archive_type == 'dmg':
        extract_file_from_dmg(
            archive_path, extract_path, dst, os.path.join(relative_file_path)
        )
    elif archive_type == 'tar':
        extract_file_from_tar(archive_path, extract_path, dst, relative_file_path)
    else:
        raise ValueError(
            f'cannot extract from {archive_path}'
            ' unsupported extension {archive_type}'
        )


def extract_file_from_dmg(dmg_path, mount_path, dst, relative_file_path):
    cdr_path = os.path.join('/tmp', os.path.basename(dmg_path).split('.')[0] + '.cdr')

    subprocess.run(['hdiutil', 'convert', dmg_path, '-format', 'UDTO', '-o', cdr_path])

    subprocess.run(
        ['hdiutil', 'attach', '-mountpoint', mount_path, cdr_path], text=True, input="Y"
    )
    try:
        copy_matching_paths(relative_file_path, dst, root_dir=mount_path)
    finally:
        subprocess.run(['hdiutil', 'detach', mount_path])


def extract_file_from_tar(tar_path, extract_path, dst, relative_file_path):
    with tarfile.open(tar_path) as tar_file:
        tar_file.extractall(extract_path)

    copy_matching_paths(os.path.join(extract_path, relative_file_path), dst)


def extract_tar(tar_path, dst):
    with tarfile.open(tar_path) as tar_file:
        tar_file.extractall(dst)
