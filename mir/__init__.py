'''MIR library in python'''

# todo: consider importing from modules in all init

import os

def get_resource_path(name=None):
    path = os.path.realpath(__file__)
    directory = os.path.dirname(path)
    subdirectory = os.path.join(directory, "resources")
    if name is None:
        filenames = os.listdir(subdirectory)
        return sorted(filenames)
    path = os.path.join(subdirectory, name)
    if not (os.path.isfile(path) or os.path.isdir(path)):
        raise Exception("Missing resource")
    return path