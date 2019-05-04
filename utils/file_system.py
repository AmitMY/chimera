import errno
import os
import shutil
import tempfile
from os.path import join, isfile
from typing import List


def makedir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def rmdir(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def rmfile(path):
    os.remove(path)


def listdir(directory, full=True):
    files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
    if full:
        files = [join(directory, f) for f in files]
    return files


def copyfile(src, dst):
    return shutil.copyfile(src, dst)


def temp_name(suffix=None):
    fd, path = tempfile.mkstemp(suffix)
    os.close(fd)  # Close connection so no too-many-connections
    return path


def temp_dir():
    path = tempfile.mkdtemp()
    return path + os.sep


def save_temp(rows: List[str]):
    name = temp_name()
    f = open(name, "w", encoding="utf-8")
    f.write("\n".join(rows))
    f.close()
    return name


def save_temp_bin(bin: bytearray):
    name = temp_name()
    f = open(name, "wb")
    f.write(bin)
    f.close()
    return name
