import project_config as config
from module_minc_keras.minc_keras import *

PROJECT_DIR = config.get_project_path()
DATASET_DIR = PROJECT_DIR + "/datasets"
OUTPUT_DIR = PROJECT_DIR + "/output/keras_implementation"


def main():
    os.chdir(DATASET_DIR)
    setup_dirs('mri_pad_4_results')
    [images_mri_pad_4, data_mri_pad_4] = prepare_data('mri', 'mri_pad_4_results/data', 'mri_pad_4_results/report',
                                                      input_str='_T1w_anat_rsl.mnc', label_str='variant-seg',
                                                      images_fn='mri_pad_4_results/report/mri_unet.csv', pad_base=4,
                                                      clobber=True)

    # Load data
    Y_validate_mri_pad_4 = np.load(data_mri_pad_4["validate_y_fn"] + '.npy')
    nlabels_mri_pad_4 = len(np.unique(Y_validate_mri_pad_4))

    X_train_mri_pad_4 = np.load(data_mri_pad_4["train_x_fn"] + '.npy')
    Y_train_mri_pad_4 = np.load(data_mri_pad_4["train_y_fn"] + '.npy')
    X_validate_mri_pad_4 = np.load(data_mri_pad_4["validate_x_fn"] + '.npy')

    X_test_mri_pad_4 = np.load(data_mri_pad_4["test_x_fn"] + '.npy')
    Y_test_mri_pad_4 = np.load(data_mri_pad_4["test_y_fn"] + '.npy')

    Y_test_mri_pad_4 = to_categorical(Y_test_mri_pad_4)
    Y_train_mri_pad_4 = to_categorical(Y_train_mri_pad_4, num_classes=nlabels_mri_pad_4)
    Y_validate_mri_pad_4 = to_categorical(Y_validate_mri_pad_4, num_classes=nlabels_mri_pad_4)

    # if you change the number of times you downsample with max_pool,
    # then you need to rerun prepare_data() with pad_base=<number of downsample nodes>
    model_name = "model_of_unet_at_mri.hdf5"

    # Define the architecture of neural network
    IN = Input(shape=(data_mri_pad_4['image_dim'][1], data_mri_pad_4['image_dim'][2], 1))

    BN1 = BatchNormalization()(IN)

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(BN1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up5 = UpSampling2D(size=(2, 2))(conv5)
    # up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc5 = Concatenate(axis=3)([up5, conv4])
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conc5)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)
    # up6 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv6)
    conc6 = Concatenate(axis=3)([up6, conv3])
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conc6)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up7 = UpSampling2D(size=(2, 2))(conv7)
    # up7 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv7)
    conc7 = Concatenate(axis=3)([up7, conv2])
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conc7)  # (up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up8 = UpSampling2D(size=(2, 2))(conv8)
    # up8 = Conv2DTranspose( filters=512, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    conc8 = Concatenate(axis=3)([up8, conv1])
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conc8)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(nlabels_mri_pad_4, 1, 1, activation='softmax')(conv9)

    model = keras.models.Model(input=[IN], output=conv10)

    print(model.summary())

    # set compiler
    ada = keras.optimizers.Adam(0.0001)
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=ada, metrics=['acc'])
    # fit model
    history = model.fit([X_train_mri_pad_4],
                        Y_train_mri_pad_4,
                        validation_data=([X_validate_mri_pad_4], Y_validate_mri_pad_4),
                        epochs=3)
    # save model
    model.save(model_name)
    # test model
    test_score = model.evaluate(X_test_mri_pad_4, Y_test_mri_pad_4)
    print("Test :", test_score)


if __name__ == "__main__":
    main()
