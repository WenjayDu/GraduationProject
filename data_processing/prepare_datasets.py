import os
import pickle
from config_and_utils import GlobalVar
from module_minc_keras.minc_keras import setup_dirs
from module_minc_keras.prepare_data import prepare_data

PROJECT_PATH = GlobalVar.PROJECT_PATH
SOURCE_DATA_DIR = GlobalVar.DATASET_PATH + "/mri"
DATA_DIR = GlobalVar.DATASET_PATH + "/mri_pad_4_results/data"
REPORT_DIR = GlobalVar.DATASET_PATH + "/mri_pad_4_results/report"
SERIALIZE_FILE = GlobalVar.DATASET_PATH + "/mri_pad_4_results/prepare_mri_dataset_return"


def serialize_object(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def prepare_mri_dataset():
    """
    used to prepare mri dataset with padding = 4
    """
    os.chdir(PROJECT_PATH)
    if not os.path.exists(PROJECT_PATH + "/datasets/mri"):
        print("mri source data does not exist, start creating...")
        print("decompressing the output.tar.bz2 file from minc_keras project...")

        cmd = "tar -jxvf " + PROJECT_PATH + "/module_minc_keras/data/output.tar.bz2 -C " + PROJECT_PATH + "/datasets &> /dev/null"
        print("executing -> ", cmd)
        os.system(cmd)

        cmd = "mv " + PROJECT_PATH + "/datasets/output " + PROJECT_PATH + "/datasets/mri"
        print("executing -> ", cmd)
        os.system(cmd)
        print("done creating")

    setup_dirs(PROJECT_PATH + '/datasets/mri_pad_4_results')

    os.chdir(PROJECT_PATH + "/datasets")
    result = prepare_data(SOURCE_DATA_DIR, DATA_DIR, REPORT_DIR,
                          input_str='_T1w_anat_rsl.mnc',
                          label_str='variant-seg',
                          pad_base=4,
                          clobber=True)
    os.chdir(PROJECT_PATH)
    serialize_object(result, SERIALIZE_FILE)
    print("result of def prepare_data() has been writen to", SERIALIZE_FILE)
    print("\npreparing mri dataset with pad 4 done\n")


if __name__ == "__main__":
    prepare_mri_dataset()
