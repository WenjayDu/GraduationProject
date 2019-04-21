import os
import project_config as config
from module_minc_keras.minc_keras import setup_dirs
from module_minc_keras.prepare_data import prepare_data


def prepare_mri_dataset():
    """
    used to prepare mri dataset with padding = 4
    """
    current_dir_pardir = config.get_pardir_containing_file(__file__)
    print("uncompressing the output.tar.bz2 file from minc_keras project...")
    os.system("tar -jxvf " + current_dir_pardir + "/module_minc_keras/data/output.tar.bz2")
    os.system("mv ./output " + current_dir_pardir + "/datasets/mri")
    setup_dirs(current_dir_pardir + '/datasets/mri_pad_4_results')
    os.chdir(current_dir_pardir + "/datasets")
    [images_mri_pad_4, data_mri_pad_4] = prepare_data('mri', 'mri_pad_4_results/data', 'mri_pad_4_results/report',
                                                      input_str='_T1w_anat_rsl.mnc',
                                                      label_str='variant-seg',
                                                      pad_base=4,
                                                      clobber=True)
    print("\nmri dataset prepared\n")
    os.chdir(config.get_project_path())
    return [images_mri_pad_4, data_mri_pad_4]


if __name__ == "__main__":
    prepare_mri_dataset()
