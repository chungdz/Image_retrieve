from email import header
import pandas as pd
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm 
import json
from process_data.change_shape import changeImageShape
import scipy.io

#==================================================================================#
parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="root path of all data")
parser.add_argument("--filter_type", default="Gaussian", type=str,
                        help="Filter type for empty space of images after rescaling, Black, White, or Gaussian")
parser.add_argument("--image_resolution", default=224, type=int,
                        help="square image resolution after resized")
parser.add_argument("--numChannels", default=3, type=int,
                        help="number of channels of input images, for RGB images, its 3")

args = parser.parse_args()
path = os.path.join(args.dpath, "Image_data/sv_data/image/")  # image file path
print("file path", path)
files = os.listdir(path)

infopath = os.path.join(args.dpath, "Image_data/sv_data/sv_make_model_name.mat")
test_indexpath = os.path.join(args.dpath, "tindexinfo.csv")
test_set_path = os.path.join(args.dpath, "test.npy")

infomat = scipy.io.loadmat(infopath)

test_set = []
index = 0   # index for test set
for l in tqdm(files, total=len(files)):
    l1 = l + "/"
    files1 = None
    files1 = os.listdir(path + l1)
    
    for imagep in files1:
        # find correspond carmodel number
        carmodel_number = infomat['sv_make_model_name'][int(l) - 1][2].flatten().tolist()[0]
        # add image into matrix and testset
        temp_set = [os.path.join(l, imagep), index, carmodel_number]
        test_set.append(temp_set)
        index += 1

test_index_info = pd.DataFrame(test_set, columns=["Path", "Index", "Class"])
test_index_info.to_csv(test_indexpath, index=None)
testnp = test_index_info[["Index", "Class"]].values
np.save(test_set_path, testnp)
print(test_index_info.shape, testnp.shape)

