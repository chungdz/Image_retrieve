import pandas as pd
import argparse
import os
from tqdm import tqdm
import pickle
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="cifar100", type=str,
                        help="root path of all data")
args = parser.parse_args()

tri_path = os.path.join(args.dpath, "train_info.csv")
tei_path = os.path.join(args.dpath, "test_info.csv")
trs_path = os.path.join(args.dpath, "train_image_set.npy")
tes_path = os.path.join(args.dpath, "test_image_set.npy")

train = pickle.load(open('cifar100/train', 'rb'), encoding='bytes')
test = pickle.load(open('cifar100/test', 'rb'), encoding='bytes')

train_set = []
train_info = []
for index, (label, idata) in tqdm(enumerate(zip(train[b'fine_labels'], train[b'data']))):
    original_data = idata.reshape(3, 32, 32).transpose(1, 2, 0)
    rescaled_data = cv2.resize(original_data, (64, 64), interpolation=cv2.INTER_LINEAR)
    train_set.append(rescaled_data)
    train_info.append([index, label])

trainnp = np.array(train_set, dtype=np.uint8).transpose(0, 3, 1, 2)
traindf = pd.DataFrame(train_info, columns=['Index', 'Class'])

test_set = []
test_info = []
for index, (label, idata) in tqdm(enumerate(zip(test[b'fine_labels'], test[b'data']))):
    original_data = idata.reshape(3, 32, 32).transpose(1, 2, 0)
    rescaled_data = cv2.resize(original_data, (64, 64), interpolation=cv2.INTER_LINEAR)
    test_set.append(rescaled_data)
    test_info.append([index, label])

testnp = np.array(test_set, dtype=np.uint8).transpose(0, 3, 1, 2)
testdf = pd.DataFrame(test_info, columns=['Index', 'Class'])

np.save(trs_path, trainnp)
traindf.to_csv(tri_path, index=None)
np.save(tes_path, testnp)
testdf.to_csv(tei_path, index=None)
