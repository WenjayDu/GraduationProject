# Extracted Images

Images here are extracted from minc files (file name suffix is `.mnc`), which can be obtained if you unzip the file `data/output.tar.bz2` in module minc_keras, using functions in `data_processing/extract_img_from_minc_file.py`.

There are 2 folds and 1 png file here: 

### sub-00031_task-01_ses-01_T1w_anat_rsl 
This fold contains images in `sub-00031/sub-00031_task-01_ses-01_T1w_anat_rsl.mnc` which can be used as training features.

### sub-00031_task-01_ses-01_T1w_variant-seg_rsl
This fold contains images in `sub-00031/sub-00031_task-01_ses-01_T1w_variant-seg_rsl.mnc` which can be used as training labels.

### predictions_of_sub-00031_task-01_ses-01_T1w_anat_rsl.png 
This image is the prediction made by the model based on images in fold `sub-00031_task-01_ses-01_T1w_anat_rsl`. In each sub plot, there are 3 brain images that are the training feature, the training label and the prediction, from left to right.