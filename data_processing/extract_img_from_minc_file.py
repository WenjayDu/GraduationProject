import os
import numpy as np
from keras.preprocessing import image
from module_minc_keras.utils import safe_h5py_open
from config_and_utils import GlobalVar, get_sorted_files

SAVE_PATH = GlobalVar.DATASET_PATH + "/examples/extracted_images"


def extract_img(minc_file=None, save_path=None):
    """
    This function is used to extract images from a minc file in "mri" dir
        which contains minc files of the keras_minc project

    :param minc_file: the path of the minc file
    :param save_path: the path to save images
    :return:
    """

    if minc_file is None:
        print("U should give out the path of a minc file")
        return 1
    if save_path is None:
        save_path = SAVE_PATH
    filename = os.path.basename(minc_file)
    dir_name = filename.split('.')[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dir = os.path.join(SAVE_PATH, dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    minc_file = safe_h5py_open(minc_file, 'r')
    images = np.array(minc_file["minc-2.0"]["image"]["0"]["image"])
    images = images.reshape(list(images.shape) + [1])
    total_quantity = images.shape[0]
    for i in range(0, total_quantity):
        data = images[i]
        filename = save_dir + '/' + str(i) + '.png'
        image.save_img(filename, data)
    minc_file.close()
    print("🚩Extracted images have been saved to " + save_dir)


def create_gif(gif_name, dir_path, duration=0.25):
    """
    this function is used to generate a gif file including all images in a dir

    :param gif_name: the name of the gif file to be generated
    :param dir_path: the path of the dir containing images used to generate gif file
    :param duration: time duration between each image in the gif file
    :return:
    """
    import imageio
    if gif_name.split(".")[-1] is not "gif":
        gif_name = gif_name + ".gif"
    frames = []
    image_list = get_sorted_files(dir_path, "png")
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    print("🚩gif file ./" + gif_name + " generated successfully")
