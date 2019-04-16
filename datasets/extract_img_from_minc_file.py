import os
import cv2
from module_minc_keras.utils import safe_h5py_open

SAVE_PATH = "extracted_images"


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
    total_quantity = minc_file["minc-2.0"]["image"]["0"]["image"].shape[0]
    for i in range(0, total_quantity):
        data = minc_file["minc-2.0"]["image"]["0"]["image"][i]
        file = save_dir + '/' + str(i) + '.png'
        cv2.imwrite(file, data)
    minc_file.close()
    print("images have been saved to ./" + save_dir)


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
    png_files = os.listdir(dir_path)
    png_files.sort(key=lambda x: int(x[:-4]))
    image_list = [os.path.join(dir_path, f) for f in png_files]
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    print("generated successfully")
