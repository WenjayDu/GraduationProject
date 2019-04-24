import os
import project_config as config
from module_minc_keras.minc_keras import setup_dirs
from module_minc_keras.prepare_data import prepare_data


def prepare_mri_dataset():
    """
    used to prepare mri dataset with padding = 4
    """
    project_path = config.get_project_path()
    os.chdir(project_path)
    if not os.path.exists(project_path + "/datasets/mri"):
        print("decompressing the output.tar.bz2 file from minc_keras project...")

        cmd = "tar -jxvf " + project_path + "/module_minc_keras/data/output.tar.bz2 -C " + project_path + "/datasets &> /dev/null"
        print("executing -> ", cmd)
        os.system(cmd)

        cmd = "mv " + project_path + "/datasets/output " + project_path + "/datasets/mri"
        print("executing -> ", cmd)
        os.system(cmd)

        print("done decompressing")

    setup_dirs(project_path + '/datasets/mri_pad_4_results')

    os.chdir(project_path + "/datasets")
    [images_mri_pad_4, data_mri_pad_4] = prepare_data('mri', 'mri_pad_4_results/data', 'mri_pad_4_results/report',
                                                      input_str='_T1w_anat_rsl.mnc',
                                                      label_str='variant-seg',
                                                      pad_base=4,
                                                      clobber=True)
    os.chdir(project_path)
    print("\nmri dataset prepared\n")

    return [images_mri_pad_4, data_mri_pad_4]


if __name__ == "__main__":
    prepare_mri_dataset()
