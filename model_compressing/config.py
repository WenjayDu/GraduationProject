import os
import numpy as np


class GlobalPath:
    PROJECT_PATH = os.path.split(os.path.realpath(__file__))[0].split("/module_pocketflow")[0]
    DATASET_PATH = PROJECT_PATH + "/datasets"
    OUTPUT_PATH = PROJECT_PATH + "/output"


def cal_np_unique_num(file_path):
    return len(np.unique(np.load(file_path)))
