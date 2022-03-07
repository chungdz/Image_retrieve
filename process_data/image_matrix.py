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
def generateImageSet(dataframe, index=None, start=None, end=None, path=None, 
                     filter_type = "Gaussian", res=224, numChannels=3, sigma=1000, filter_size=7):
    image_set = []
    # index_set = []
    # model_set = []
    #get image one by one:
    if index is None:
        index = 0
    if start is None:
        start = 0
    if end is None:
        end = dataframe.shape[0]

    for image_num in trange(start, end):
        #generate 10 sub image for one image
        assert(image_num == dataframe['Index'][image_num])
        impath = dataframe["Path"].iloc[image_num]
        if not path is None:
            impath = path + dataframe["Path"].iloc[image_num]
        resized_image = changeImageShape(impath, res, numChannels, filter_type, sigma, filter_size)
        if resized_image.sum() == 0:
            exit()
        image_set.append(resized_image)
        # index_set.append(index)
        # model_set.append(dataframe["Class"].iloc[image_num])
        index += 1
        if index > 2000:
            break
        # print("{}: Completed".format(index))
    imageset = np.array(image_set, dtype = np.uint8)

    #save index info as a dataframe
    # data = {'Index': index_set, 'Class': model_set}

    # index_df = pd.DataFrame(data)
    # return imageset, index_df
    return imageset

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="root path of all data")
parser.add_argument("--image_info", default="car_front.csv", type=str,
                        help="csv file name with path and class information as two columns for all images")
parser.add_argument("--image_root_path", default="Image_data/data/image/", type=str,
                        help="dpath + image_root_path + path_info_in_csv_file = absolute path of each image")
parser.add_argument("--mname", default='imageset.npy', type=str,
                        help="matrix data file name")
# parser.add_argument("--iname", default='indexinfo.csv', type=str,
#                         help="index and class information csv used by following file")
parser.add_argument("--filter_type", default="Gaussian", type=str,
                        help="Filter type for empty space of images after rescaling, Black, White, or Gaussian")
parser.add_argument("--image_resolution", default=224, type=int,
                        help="square image resolution after resized")
parser.add_argument("--numChannels", default=3, type=int,
                        help="number of channels of input images, for RGB images, its 3")
args = parser.parse_args()
matrix_path = os.path.join(args.dpath, args.mname)
# image_index = os.path.join(args.dpath, args.iname)
front_csv_path = os.path.join(args.dpath, args.image_info)
img_path = os.path.join(args.dpath, args.image_root_path)

# npm, dfm = generateImageSet(pd.read_csv(front_csv_path), path=img_path, filter_type=args.filter_type, res=args.image_resolution, numChannels=args.numChannels)
npm = generateImageSet(pd.read_csv(front_csv_path), path=img_path, filter_type=args.filter_type, res=args.image_resolution, numChannels=args.numChannels)
npm = np.transpose(npm, (0, 3, 1, 2))
np.save(matrix_path, npm)
print(npm.shape)
# dfm.to_csv(image_index, index=None)


