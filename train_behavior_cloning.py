import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.models import Sequential
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_lines():
    lines = []
    with open("../data/driving_log.csv") as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines

correction = 0.2


def sample_generator(samples, batch_size):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    path = batch_sample[i]
                    file = path.split('/')[-1]
                    path = '../data/IMG/' + file
                    angle = float(batch_sample[3])
                    if i == 1:
                        angle += correction
                    if i == 2:
                        angle -= correction
                    angles.append(angle)
                    img = cv2.imread(path)
                    images.append(img)
                    images.append(np.fliplr(img))
                    angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def basic():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(1))
    return model


def nvidia():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Conv2D(24,5,5,activation='relu'))
    model.add(Conv2D(36,5,5,activation='relu'))
    model.add(Conv2D(48,5,5,activation='relu'))
    model.add(Conv2D(64,3,3,activation='relu'))
    model.add(Conv2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

samples = load_lines()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = sample_generator(train_samples, batch_size=32)
validation_generator = sample_generator(validation_samples, batch_size=32)

model = basic()

model.compile(optimizer='adam', loss='mse')

history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
        validation_data=validation_generator, nb_val_samples=len(validation_samples),
        nb_epoch=10)

model.save('model.h5')
