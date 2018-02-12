import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


lines = []

with open("./data_old/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    # for index in range(3):
    source_image = line[0]
    filename = source_image.split('/')[-1]
    current_path = "./data_old/IMG/" +filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Preprocess data
# X_normalized = np.array((X_train / 255.0) - 0.5 )



model = Sequential()
model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.summary()

model.save('model.h5')
exit()