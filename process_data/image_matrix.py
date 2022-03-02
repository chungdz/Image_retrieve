from unittest.mock import NonCallableMagicMock
import cv2
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm, trange
from process_data.change_shape import changeImageShape



#============================================================================#
#generateImageSet(dataframe, index, start, end)
#Input: 
#   dataframe: dataframe contain all information about each car
#   index: start index 
#   start: start image number from the dataframe
#   end:   end image number from the dataframe
#
#Output: 
#   imageset:   One big matrix contain all 3X224X224 images
#   index_df:   dataframe contain image index and correspond car model info
#============================================================================#
def generateImageSet(dataframe, index=None, start=None, end=None, path=None, filter_type = "Gaussian"):
    image_set = []
    index_set = []
    model_set = []
    #get image one by one:
    if index is None:
        index = 0
    if start is None:
        start = 0
    if end is None:
        end = dataframe.shape[0]

    for image_num in trange(start, end):
        #generate 10 sub image for one image
        impath = dataframe["Path"].iloc[image_num]
        if not path is None:
            impath = path + dataframe["Path"].iloc[image_num]
        resized_image = changeImageShape(impath, res=224, numChannels=3, filter_type, sigma=1000, filter_size=7)
        if resized_image.sum() == 0:
            exit()
        image_set.append(resized_image)
        index_set.append(index)
        model_set.append(dataframe["CarModel"].iloc[image_num])
        index+=1
        # print("{}: Completed".format(index))
    imageset = np.array(image_set, dtype = np.uint8)

    #save index info as a dataframe
    data = {'Index': index_set, 'Carmodel': model_set}

    index_df = pd.DataFrame(data)
    return imageset, index_df

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--filter_type", default="Gaussian", type=str,
                        help="Filter type for empty space of images")
args = parser.parse_args()
matrix_path = os.path.join(args.dpath, 'imageset.npy')
image_index = os.path.join(args.dpath, 'indexinfo.csv')
front_csv_path = os.path.join(args.dpath, "cat_front.csv")
img_path = os.path.join(args.dpath, "Image_data/data/image/")

if args.filter_type == "Gaussian":
    npm, dfm = generateImageSet(pd.read_csv(front_csv_path), path=img_path)
elif args.filter_type == "Black":
    npm, dfm = generateImageSet(pd.read_csv(front_csv_path), path=img_path,filter_type ="Black")

np.save(matrix_path, np.transpose(npm, (0, 3, 1, 2)))
dfm.to_csv(image_index, index=None)


