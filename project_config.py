import os


class GlobalVar:
    """
    all global variables are set here
    """
    project_path = os.path.split(os.path.realpath(__file__))[0]


def get_project_path():
    return GlobalVar.project_path


def get_dir_containing_file(file):
    """
    :param file: specified file, __file__ is ok
    :return: the absolute path of the dir containing the specified file
    """
    return os.path.split(os.path.realpath(file))[0]


def get_pardir_containing_file(file):
    """
    :param file: specified file, __file__ is ok
    :return: the absolute path of the parent dir of the dir containing the specified file
    """
    return os.path.abspath(os.path.join(os.path.dirname(file), os.path.pardir))
