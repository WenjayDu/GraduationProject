import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(asctime)s: %(message)s')


# Global variables
class GlobalVar:
    """
    all global variables are set here
    """
    PROJECT_PATH = os.path.split(os.path.realpath(__file__))[0]
    DATASET_PATH = PROJECT_PATH + "/datasets"
    OUTPUT_PATH = PROJECT_PATH + "/output"


# commonly used functions are also placed here
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


def cal_np_unique_num(file_path):
    return len(np.unique(np.load(file_path)))
