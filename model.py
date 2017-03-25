import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import argparse

def load_samples(dir):
    lines = []
    with open(dir+"/driving_log.csv") as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines


class SampleGenerator:
    correction = 0.2

    def __init__(self, dir, samples, batch_size=16, flip=False, sides=False):
        self.dir = dir
        self.samples = samples
        self.batch_size = batch_size
        self.flip = flip
        self.sides = sides

    def __len__(self):
        l = len(self.samples)
        if self.flip: l *= 2
        if self.sides: l *= 3
        return l

    def generate(self):
        num_samples = len(self.samples)
        while True:
            shuffle(samples)
            for offset in range(0, num_samples, self.batch_size):
                batch_samples = self.samples[offset:offset+self.batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    for i in range(3):
                        if not self.sides and i>0: break
                        path = batch_sample[i]
                        file = path.split('/')[-1]
                        path = self.dir+'/IMG/' + file
                        angle = float(batch_sample[3])
                        if i == 1:
                            angle += SampleGenerator.correction
                        if i == 2:
                            angle -= SampleGenerator.correction

                        angles.append(angle)
                        img = cv2.imread(path)
                        images.append(img)
                        if self.flip:
                            images.append(np.fliplr(img))
                            angles.append(-angle)

                X = np.array(images)
                y = np.array(angles)
                yield shuffle(X, y)


class ModelBuilder:
    @staticmethod
    def build(model):
        return { "basic": ModelBuilder.basic_model,
                 "nvidia": ModelBuilder.nvidia_model }.get(model)()

    @staticmethod
    def basic_model():
        print("building basic model")
        model = Sequential()
        model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(1))
        return model

    @staticmethod
    def nvidia_model():
        print("building nvidia model")
        model = Sequential()
        model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        model.add(Conv2D(24,5,5, activation='relu', subsample=(2,2)))
        model.add(Conv2D(36,5,5, activation='relu', subsample=(2,2)))
        model.add(Conv2D(48,5,5, activation='relu', subsample=(2,2)))
        model.add(Conv2D(64,3,3, activation='relu'))
        model.add(Conv2D(64,3,3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-model', type=str, default="basic")
parser.add_argument('--flip', default=False, action='store_true')
parser.add_argument('--sides', default=False, action='store_true')
args = parser.parse_args()

samples = load_samples(args.dir)
train_samples, validation_samples = train_test_split(samples, test_size=0.25)

train_generator = SampleGenerator(args.dir, train_samples, flip=args.flip, sides=args.sides)
valid_generator = SampleGenerator(args.dir, validation_samples, flip=args.flip, sides=args.sides)


model = ModelBuilder.build(args.model)

model.compile(optimizer='adam', loss='mse')

checkpointer = ModelCheckpoint(filepath="model.hd5", verbose=1, save_best_only=True)

history = model.fit_generator(train_generator.generate(), samples_per_epoch=len(train_generator),
                              validation_data=valid_generator.generate(), nb_val_samples=len(valid_generator),
                              nb_epoch=args.epochs, callbacks=[checkpointer])
