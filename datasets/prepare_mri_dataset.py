import os
from module_minc_keras.prepare_data import prepare_data
from module_minc_keras.minc_keras import setup_dirs


def prepare():
    os.system("tar -jxvf ../module_minc_keras/data/ouput.tar.bz2 &> /dev/null "
              "&& mv ../module_minc_keras/data/output ./mri")
    setup_dirs('mri_pad_4_results')
    [images_mri_pad_4, data_mri_pad_4] = prepare_data('mri', 'mri_pad_4_results/data', 'mri_pad_4_results/report',
                                                      input_str='_T1w_anat_rsl.mnc',
                                                      label_str='variant-seg',
                                                      pad_base=4,
                                                      clobber=True)
    print("\ndataset prepared\n")
    return [images_mri_pad_4, data_mri_pad_4]


if __name__ == "__main__":
    prepare()
