# data_processing dir

## About `convert_npy_to_tfreocrds.py`
Can be used to convert .npy files to .tfrecords files.

## About `prepare_dataset.py`
Run `prepare_dataset.py` first to help prepare the dataset.

## About `extract_img_from_minc_file.py`
It is noteworthy that `extract_img_from_minc_file.py` is created to extract images from a minc file and create a gif with them.

The `extracted_images` in `../datasets/examples` is an example dir, which contains images from `sub-00031/sub-00031_task-01_ses-01_T1w_anat_rsl.mnc` that is a file in mri data from minc_keras project, besides, there is a gif file created with these .png files as well.

If `prepare_dataset.py` is executed, the `mri` dir can be seen in the `../datasets`, and `sub-00031/sub-00031_task-01_ses-01_T1w_anat_rsl.mnc` also can be found under this folder.

