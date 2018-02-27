import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import optimizers
import tensorflow as tf


def generator(images_path, samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for index in range(3):
                    current_path = images_path + batch_sample[index].split('/')[-1]
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    steering_angle = float(batch_sample[3])
                    correction = 0.20
                    if index == 1: #left
                        steering_angle += correction
                    if index == 2:
                        steering_angle -= correction

                    images.append(image)
                    angles.append(steering_angle)

                    images.append(cv2.flip(image, 1))
                    angles.append(steering_angle * -1)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def load_generator_images_and_measures(csv_path, images_path):
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(images_path, train_samples, batch_size=64)
    validation_generator = generator(images_path, validation_samples, batch_size=64)

    samples_per_epoch = len(train_samples)*3*2
    nb_val_samples = len(validation_samples)*3*2

    return train_generator, validation_generator,samples_per_epoch, nb_val_samples


def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def trainAndSave(model, train_generator, validation_generator, samples_per_epoch, nb_val_samples, modelFile, epochs = 3):

    # adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
                        samples_per_epoch=samples_per_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=nb_val_samples,
                        nb_epoch=epochs,
                        verbose=1)

    model.save(modelFile)


train_generator, validation_generator,\
    samples_per_epoch, nb_val_samples = load_generator_images_and_measures("./data/driving_log.csv", "./data/IMG/")
model = nvidia_model()
trainAndSave(model, train_generator, validation_generator, samples_per_epoch, nb_val_samples, 'model.h5')
