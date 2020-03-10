from glob import glob
import os


def get_files_from_dir(path, extension):
    assert os.path.exists(path), "Path not found"
    return sorted(glob(os.path.join(path, "*.{}".format(extension))))
