import csv
import cv2
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import argparse
from tqdm import tqdm

def load_samples(dir):
    lines = []
    with open(dir+"/driving_log.csv") as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines


def sample_angles(samples):
    angles = []
    for sample in samples:
        angle = float(sample[3])
        angles.append(angle)
    return angles


def stratify_samples(samples):
    angles = sample_angles(samples)

    counts, bins = np.histogram(angles, bins=10)
    print("before stratifing", counts)

    in_bin = np.digitize(angles, bins)
    stratified_samples = []
    target_num = (500 *np.array([ 0, .1, .2, .2, 1,1,1,1,.2,.2,.1,.1])).astype(np.int32)
    #target_num = (500 *np.ones(12)).astype(np.int32)
    for bin1 in range(1,len(bins)+1):
        bin_idx = np.where(in_bin == bin1)[0]
        if len(bin_idx) == 0: continue

        bin_idx = resample(bin_idx, n_samples=target_num[bin1])
        for idx1 in bin_idx:
            stratified_samples.append(samples[idx1])

    angles = sample_angles(stratified_samples)
    counts, bins = np.histogram(angles, bins=10)
    print("after stratifing", counts)

    return stratified_samples

def preprocess_tiny(path, samples, correction=0.25, flip=False, sides=False):
    images = []
    angles = []
    with tqdm(samples, desc='preprocessing') as pbar:
        for sample in pbar:
            for i in range(3):
                if not sides and i>0: break
                file = sample[i]
                file = file.split('/')[-1]
                file = path +'/IMG/'+ file
                angle = float(sample[3])
                if i == 1:
                    angle += correction
                if i == 2:
                    angle -= correction

                angles.append(angle)
                img = cv2.imread(file)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img = cv2.resize(img,(32,16))
                img = img[:, :, 1:2]
                images.append(img)
                if flip:
                    images.append(np.fliplr(img))
                    angles.append(-angle)

    X = np.array(images)
    y = np.array(angles)
    return shuffle(X, y)



def tiny_model(opts):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(16,32,1)))
    model.add(Conv2D(2,3,3, activation='relu'))
    model.add(MaxPooling2D((4, 4), (4, 4))),
    if opts.dropout>0: model.add(Dropout(opts.dropout))
    model.add(Flatten())
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

    model = tiny_model(args)
    model.compile(optimizer='adam', loss='mse')

    checkpointer = ModelCheckpoint(filepath="tiny.hd5", verbose=1, save_best_only=True)

    if args.stratify:
        samples = stratify_samples(samples)

    if args.resample:
        samples = resample(samples, n_samples=args.resample)

    X, y = preprocess_tiny(args.dir, samples, correction=args.correction, flip=args.flip, sides=args.sides)
    train_X, validation_X, train_y, validation_y = train_test_split(X, y, test_size=args.split)

    model.fit(train_X, train_y, validation_data=(validation_X,validation_y),
              batch_size=args.batch, nb_epoch=args.epochs, shuffle=True, callbacks=[checkpointer])
