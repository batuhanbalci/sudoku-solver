import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from keras.models import load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import image_parsing as img_parsing
import solver
import requests


def slice_digits(img):
    cv2.imshow('imagee', img)

    puzzle = []
    divisor = img.shape[0] // 9
    for i in range(9):
        row = []
        for j in range(9):
            # slice image, reshape it to 28x28 (mnist reader size)
            row.append(cv2.resize(img[i * divisor:(i + 1) * divisor,
                                  j * divisor:(j + 1) * divisor][3:-3, 3:-3],
                                  dsize=(28, 28),
                                  interpolation=cv2.INTER_CUBIC))
        puzzle.append(row)

    #cv2.imshow('imagee', puzzle[3][4])


    #buraya kadar doğru puzzle arr'i


    #model = tf.keras.models.Sequential()
    """"
    model.add(tf.keras.layers.Conv2D(254, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3)))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(140, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
    """

    model_1_filter_size = 3
    epochs = 10

    #model = tf.keras.models.model_from_json()
    #sequantial dan json a çevirdim
    """""
    model = tf.keras.models.Sequential(
                        [Convolution2D(filters=64, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu', input_shape=(28, 28, 1)),
                         Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.5),
                         Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'),
                         Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.5),
                         Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'),
                         MaxPooling2D(pool_size=(2, 2)),
                         Dropout(0.5),
                         Flatten(),
                         Dense(1024, activation='elu'),
                         Dropout(0.5),
                         Dense(1024, activation='elu'),
                         Dropout(0.5),
                         Dense(10, activation='softmax'),
                         ])
    """

    """
    model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'))
    model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=(model_1_filter_size, model_1_filter_size),
                                       padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='elu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1024, activation='elu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    """

    """""
    model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, batch_size=256, epochs=epochs, shuffle=True, verbose=1,
               validation_data=(test_images, test_labels))
               """

    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

    f = open(desktop + '/258epochs_model_7.json', 'r')
    model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(desktop + '/258epochs_model_7.h5')


    count = 0
    for row in puzzle:
        for spot in row:
            if np.mean(spot) > 6:
                count += 1
                #plt.imshow(spot)
                #plt.show()
                #print(model.predict_classes(spot.reshape(1, 1, 28, 28).astype('float32')/255))
                #print(model.predict_classes(spot.reshape(1,28,28,1).astype('float32')/255))

    print(count, ' digits are recognized')

    template = [
        [0 for _ in range(9)] for _ in range(9)
    ]

    for i, row in enumerate(puzzle):
        for j, spot in enumerate(row):
            if np.mean(spot) > 6:
                template[i][j] = model.predict_classes(spot.reshape(1, 1, 28, 28).astype('float32') / 255)[0]

    print(template)

    """
    flat_list = np.asarray(puzzle[0][0])
    flat_list = flat_list.ravel()
    flat_list = str(list(flat_list))
    flat_list = flat_list.replace("\\r\\n", "")

    input_data = "{\"data\": [" + flat_list + "]}"
    resp = requests.post("http://abc25a52-54ea-49fa-bfb5-e93e7d6b135b.eastus.azurecontainer.io/score", data=input_data,
                         headers={'Content-Type': 'application/json'})
    print('Azure : ', resp.text)
    """

    if solver.validSolvedPuzzle(template)==True:
        print("Doğru")
    else:
        print("Yanlış")

    solver.board = template
    solver.solve(template)
    solver.print_board(template)

    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    #cv2.destroyAllWindows()
    # Close all windows
    return template


def parse_grid(path):
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = img_parsing.pre_process_image(original)
    corners = img_parsing.find_corners_of_largest_polygon(processed)
    cropped = img_parsing.crop_and_warp(original, corners)
    squares = img_parsing.infer_grid(cropped)
    digits = img_parsing.get_digits(cropped, squares, 28)
    img = img_parsing.show_digits(digits)
    slice_digits(img)


def main():
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    parse_grid(desktop + '/Image-of-a-Sudoku-puzzle-from-our-dataset2.png')


if __name__ == '__main__':
    main()

