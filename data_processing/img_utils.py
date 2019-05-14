import os
import numpy as np
from keras.preprocessing import image
from module_minc_keras.utils import safe_h5py_open
import matplotlib.pyplot as plt
from config_and_utils import GlobalVar, get_sorted_files, logging
from module_minc_keras.utils import normalize

SAVE_PATH = GlobalVar.DATASET_PATH + "/examples/extracted_images"


def show_argmax_result(img, argmax_axis=2):
    """
    convert an img to a hot colormap
    :param img: path of an img to be converted or an numpy array
    :param save_path: save path
    :param argmax_axis:
    :return:
    """
    if type(img) == str:
        img = image.load_img(img)
        img.show()
        img = image.img_to_array(img)
    img = np.argmax(img, axis=argmax_axis)
    img = np.expand_dims(img, axis=argmax_axis)
    img = image.array_to_img(img)
    img.show()


def to_hot_cmap(img, save_path=None):
    """
    convert an img to a hot colormap
    :param img: path of an img or a dir containing png images to be converted or an numpy array
    :param save_path: save path
    :param argmax_axis:
    :return:
    """
    if type(img) == str:
        if os.path.isfile(img):
            img = image.load_img(img)
            img = image.img_to_array(img)
            img = np.argmax(img, axis=2)
        elif os.path.isdir(img):
            logging.info("🚩️Converting all .png files in " + img)
            image_list = get_sorted_files(img, "png")
            example = image.load_img(image_list[0])
            example = image.img_to_array(example)
            fig = plt.figure(frameon=False,
                             figsize=(example.shape[1] / 500, example.shape[0] / 500),  # figsize(width, height)
                             dpi=500)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if not os.path.exists(img + "/to_hot_cmap"):
                os.makedirs(img + "/to_hot_cmap")
            for i in image_list:
                save_path = img + "/to_hot_cmap/" + i.split('/')[-1]
                img_arr = image.load_img(i)
                img_arr = np.argmax(img_arr, axis=2)
                img_arr = image.img_to_array(img_arr)
                img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1])
                img_arr = normalize(img_arr)
                ax.imshow(img_arr, cmap="hot")
                fig.savefig(save_path)
            logging.info("🚩️Done converting, converted images are saved to " + img + "/to_hot_cmap/")
            plt.close()
            return 0
    if len(list(img.shape)) == 4:  # this means the img is a tensor, in which the 1st dim is the num of samples
        img = np.argmax(img, axis=3)
        img = img.reshape(list(img.shape[1:3]))
    else:
        img = np.argmax(img, axis=2)
    img = normalize(img)
    fig = plt.figure(frameon=False,
                     figsize=(img.shape[1] / 500, img.shape[0] / 500),  # figsize(width, height)
                     dpi=500)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap="hot")
    if save_path is None:
        save_path = "to_hot_cmap.png"
    fig.savefig(save_path)
    logging.info("🚩Saved prediction to " + save_path)
    plt.close()
    del img


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
