import os
import pickle
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config_and_utils import GlobalVar
from module_minc_keras.minc_keras import setup_dirs
from module_minc_keras.prepare_data import prepare_data

PROJECT_PATH = GlobalVar.PROJECT_PATH
DATASET_PATH = GlobalVar.DATASET_PATH


def serialize_object(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def prepare_dataset(dataset_name="mri", ratios=[0.75, 0.15]):
    """
    used to prepare dataset with padding = 4
    :param dataset_name: name of dataset, "mri" or "sorteo", default "mri"
    :param ratios: a list containing 2 elements, the ratio of train and validation dataset respectively
    :return:
    """
    if not (dataset_name == "mri" or dataset_name == "sorteo"):
        exit("❗️Error: dataset_name can only be 'mri' or 'sorteo'")
    dataset_name = "/" + dataset_name
    os.chdir(DATASET_PATH)
    if not os.path.exists(DATASET_PATH + dataset_name):
        print("❗️" + dataset_name + " source data does not exist, start creating...")
        try:
            if dataset_name == "/mri":
                print("🚩decompressing the output.tar.bz2 file from minc_keras project...")
                cmd = "tar -jxvf " + PROJECT_PATH + "/module_minc_keras/data/output.tar.bz2 -C " + PROJECT_PATH + "/datasets &> /dev/null"
                print("🚩executing -> ", cmd)
                os.system(cmd)
                cmd = "mv " + DATASET_PATH + "/output " + DATASET_PATH + "/mri"
                print("🚩executing -> ", cmd)
                os.system(cmd)
            elif dataset_name == "/sorteo":
                print("🚩downloading sorteo dataset...")
                cmd = "wget https://amnesia.cbrain.mcgill.ca/deeplearning/sorteo.tar.bz2 --no-check-certificate"
                print("🚩executing -> ", cmd)
                os.system(cmd)
                cmd = "mkdir sorteo && tar -jxvf sorteo.tar.bz2 -C sorteo &> /dev/null"
                print("🚩executing -> ", cmd)
                os.system(cmd)
            print("done creating")
        except Exception:
            exit("❗️some error happened, please try again")
    # make dirs to store generated dataset
    path_to_store_result = DATASET_PATH + dataset_name + '_pad_4_results'
    setup_dirs(path_to_store_result)

    source_data_dir = DATASET_PATH + dataset_name
    data_dir = path_to_store_result + "/data"
    report_dir = path_to_store_result + "/report"
    serialized_file = path_to_store_result + "/serialized_dataset_object"
    input_str, label_str = None, None
    if dataset_name == "/mri":
        input_str = '_T1w_anat_rsl.mnc'
        label_str = 'variant-seg'
    elif dataset_name == "/sorteo":
        input_str = '_pet.mnc'
        label_str = 'brainmask'
    result = prepare_data(source_data_dir, data_dir, report_dir,
                          input_str=input_str,
                          label_str=label_str,
                          pad_base=4,
                          clobber=True,
                          ratios=ratios)
    os.chdir(PROJECT_PATH)
    serialize_object(result, serialized_file)
    print("🚩returned object of prepare_data() has been writen to", serialized_file)
    print("🚩Done preparing " + path_to_store_result + " dataset\n")


if __name__ == "__main__":
    prepare_dataset(dataset_name="mri")
    prepare_dataset(dataset_name="sorteo")
