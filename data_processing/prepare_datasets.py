import os
import pickle
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from config_and_utils import GlobalVar, logging
from module_minc_keras.minc_keras import setup_dirs
from module_minc_keras.prepare_data import prepare_data

PROJECT_PATH = GlobalVar.PROJECT_PATH
DATASET_PATH = GlobalVar.DATASET_PATH


def serialize_object(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def prepare_dataset(dataset_name="mri", ratios=[0.75, 0.15], save_path=None):
    """
    used to prepare dataset with padding = 4
    :param dataset_name: name of dataset, "mri" or "sorteo", default "mri"
    :param ratios: a list containing 2 elements, the ratio of train and validation dataset respectively
    :param save_path: save path of results, or just a dir name, which will be created under DATASET_PATH
    :return:
    """
    if not (dataset_name == "mri" or dataset_name == "sorteo"):
        sys.exit("❗️Error: dataset_name can only be 'mri' or 'sorteo'")
    dataset_name = "/" + dataset_name
    os.chdir(DATASET_PATH)
    if not os.path.exists(DATASET_PATH + dataset_name):
        logging.warning("❗️" + dataset_name + " source data does not exist, start creating...")
        try:
            if dataset_name == "/mri":
                logging.info("🚩decompressing the output.tar.bz2 file from minc_keras project...")
                cmd = "tar -jxvf " + PROJECT_PATH + "/module_minc_keras/data/output.tar.bz2 -C " + PROJECT_PATH + "/datasets &> /dev/null"
                logging.info("🚩executing -> " + cmd)
                os.system(cmd)
                cmd = "mv " + DATASET_PATH + "/output " + DATASET_PATH + "/mri"
                logging.info("🚩executing -> " + cmd)
                os.system(cmd)
            elif dataset_name == "/sorteo":
                logging.info("🚩downloading sorteo dataset...")
                cmd = "wget https://amnesia.cbrain.mcgill.ca/deeplearning/sorteo.tar.bz2 --no-check-certificate"
                logging.info("🚩executing -> " + cmd)
                os.system(cmd)
                cmd = "mkdir sorteo && tar -jxvf sorteo.tar.bz2 -C sorteo &> /dev/null"
                logging.info("🚩executing -> " + cmd)
                os.system(cmd)
            print("done creating")
        except Exception:
            sys.exit("❗️some error happened, please try again")

    # make dirs to store generated dataset
    if save_path is None:
        save_path = DATASET_PATH + dataset_name + '_pad_4'
    else:
        if save_path.__contains__("/") is False:
            save_path = DATASET_PATH + "/" + save_path
    logging.info("🚩Generated datasets will be saved to " + save_path + ", now setting up dirs...")
    setup_dirs(save_path)

    source_data_dir = DATASET_PATH + dataset_name
    data_dir = save_path + "/data"
    report_dir = save_path + "/report"
    serialized_file = save_path + "/serialized_dataset_object"
    input_str, label_str = None, None

    # if different datasets are added, tags used to diff images and labels should also be added to here
    if dataset_name == "/mri":
        input_str = '_T1w_anat_rsl.mnc'
        label_str = 'variant-seg'
    elif dataset_name == "/sorteo":
        input_str = '_pet.mnc'
        label_str = 'brainmask'

    result = prepare_data(source_data_dir, data_dir, report_dir,
                          input_str=input_str,
                          label_str=label_str,
                          pad_base=4,  # set the pad_base the same value of times you down-sample with max_pool in unet
                          clobber=True,
                          ratios=ratios)
    os.chdir(PROJECT_PATH)
    serialize_object(result, serialized_file)
    logging.info("🚩returned object of prepare_data() has been writen to " + serialized_file)
    logging.info("🚩Done preparing " + save_path + " dataset\n")


if __name__ == "__main__":
    prepare_dataset(dataset_name="mri")
    prepare_dataset(dataset_name="sorteo")
