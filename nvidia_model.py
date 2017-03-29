import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential
from sklearn.utils import shuffle, resample
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
    def __init__(self, path, samples, correction=0.25, batch_size=16, flip=False, sides=False):
        self.dir = path
        self.samples = samples
        self.batch_size = batch_size
        self.flip = flip
        self.sides = sides
        self.correction = correction

    def __len__(self):
        l = len(self.samples)
        if self.flip: l *= 2
        if self.sides: l *= 3
        return l

    def resample(self):
        return self.samples

    def generate(self):
        while True:
            samples = self.resample()
            num_samples = len(samples)
            shuffle(samples)
            for offset in range(0, num_samples, self.batch_size):
                batch_samples = samples[offset:offset+self.batch_size]
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
                            angle += self.correction
                        if i == 2:
                            angle -= self.correction

                        angles.append(angle)
                        img = cv2.imread(path)
                        images.append(img)
                        if self.flip:
                            images.append(np.fliplr(img))
                            angles.append(-angle)
                X = np.array(images)
                y = np.array(angles)
                yield shuffle(X, y)


class StratifiedSampleGenerator(SampleGenerator):
    num_bins = 10
    samples_per_bin = 500

    def __init__(self, path, samples, correction=0.25, batch_size=16, flip=False, sides=False):
        SampleGenerator.__init__(self, path, samples, correction, batch_size, flip, sides)

    def __len__(self):
        l = StratifiedSampleGenerator.samples_per_bin * StratifiedSampleGenerator.num_bins
        if self.flip: l *= 2
        if self.sides: l *= 3
        return l

    def sample_angles(self, samples):
        angles = []
        for sample in samples:
            angle = float(sample[3])
            angles.append(angle)
        return angles

    def resample(self):
        angles = self.sample_angles(self.samples)

        counts, bins = np.histogram(angles, bins=StratifiedSampleGenerator.num_bins)
        print("before stratifing", counts)

        in_bin = np.digitize(angles, bins)
        stratified_samples = []
        target_num = (500 * np.array([0, .1, .2, .2, 1, 1, 1, 1, .2, .2, .1, .1])).astype(np.int32)
        # target_num = (500 *np.ones(12)).astype(np.int32)
        for bin1 in range(1, len(bins) + 1):
            bin_idx = np.where(in_bin == bin1)[0]
            if len(bin_idx) == 0: continue

            bin_idx = resample(bin_idx, n_samples=target_num[bin1])
            for idx1 in bin_idx:
                stratified_samples.append(self.samples[idx1])

        angles = self.sample_angles(stratified_samples)
        counts, bins = np.histogram(angles, bins=StratifiedSampleGenerator.num_bins)
        print("after stratifing", counts)

        return stratified_samples


def nvidia_model(opts):
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(24,5,5, activation='relu', subsample=(2,2)))
    if opts.dropout>0: model.add(Dropout(opts.dropout))
    model.add(Conv2D(36,5,5, activation='relu', subsample=(2,2)))
    if opts.dropout>0: model.add(Dropout(opts.dropout))
    model.add(Conv2D(48,5,5, activation='relu', subsample=(2,2)))
    if opts.dropout>0: model.add(Dropout(opts.dropout))
    model.add(Conv2D(64,3,3, activation='relu'))
    if opts.dropout>0: model.add(Dropout(opts.dropout))
    model.add(Conv2D(64,3,3, activation='relu'))
    if opts.dropout>0: model.add(Dropout(opts.dropout))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-correction', type=float, default=0.25)
    parser.add_argument('-batch', type=int, default=16)
    parser.add_argument('-resample', type=int, default=0)
    parser.add_argument('-dropout', default=0, type=float)
    parser.add_argument('-split', default=0.2, type=float)
    parser.add_argument('--flip', default=False, action='store_true')
    parser.add_argument('--sides', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--stratify', default=False, action='store_true')
    args = parser.parse_args()

    samples = load_samples(args.dir)

    model = nvidia_model(args)
    model.compile(optimizer='adam', loss='mse')

    checkpointer = ModelCheckpoint(filepath="nvidia.hd5", verbose=1, save_best_only=True)

    train_samples, validation_samples = train_test_split(samples, test_size=args.split)
    if args.stratify:
        train_generator = StratifiedSampleGenerator(args.dir, train_samples,
            batch_size=args.batch, flip=args.flip, sides=args.sides, correction=args.correction)
        valid_generator = SampleGenerator(args.dir, validation_samples,
            batch_size=args.batch, flip=args.flip, sides=args.sides, correction=args.correction)
    else:
        train_generator = SampleGenerator(args.dir, train_samples,
            batch_size=args.batch, flip=args.flip, sides=args.sides, correction=args.correction)
        valid_generator = SampleGenerator(args.dir, validation_samples,
            batch_size=args.batch, flip=args.flip, sides=args.sides, correction=args.correction)

    history = model.fit_generator(train_generator.generate(), samples_per_epoch=len(train_generator),
        validation_data=valid_generator.generate(), nb_val_samples=len(valid_generator),
        nb_epoch=args.epochs, callbacks=[checkpointer])
