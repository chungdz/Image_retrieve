from unittest.mock import NonCallableMagicMock
import cv2
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm, trange
#============================================================================#
#changeImageShape(path)
#Input: the location of single image: E.g: "F:/Image_data/data/image/"
#Output: Single image with shape of 224X224X3
#
# Image first reshape scale to longer side = 224
# Full fill empty part with origin part og image after an gaussian filter
#============================================================================#
def changeImageShape(path):
    image = None                #clear image variable in case memory use error from imread()
    image  = cv2.imread(path)    
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert BGR to RGB
    if image1.shape[1] < image1.shape[0]:
        scale_percent = image1.shape[0]/224
        width = round(image1.shape[1] / scale_percent)
        height = round(image1.shape[0] / scale_percent)
        
    else:
        scale_percent = image1.shape[1]/224
        width = round(image1.shape[1] / scale_percent)
        height = round(image1.shape[0] / scale_percent)

    dim = (width, height)
    resized = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
    
    #create an empty array with size of 224*224*3:
    resized_image = np.zeros((224,224,3),dtype=np.uint8)
    
    #Copy resized image into 300*300*3 matrix
    if resized.shape[0]<224:
        center_temp = (224 - resized.shape[0])//2
        if resized.shape[0]%2:
            resized_image[center_temp+1:224-center_temp, 0:224] = resized[0:resized.shape[0], 0:224]
        else:
            resized_image[center_temp:224-center_temp, 0:224] = resized[0:resized.shape[0], 0:224]
        #Fill out blank part of image
        gaussiand_image_top = cv2.GaussianBlur(resized[0:center_temp, 0:224],(7,7),1000)
        gaussiand_image_bottom = cv2.GaussianBlur(resized[(resized.shape[0]-center_temp):resized.shape[0], 0:224],(7,7),1000)
        resized_image[0:center_temp, 0:224] = gaussiand_image_top
        resized_image[(224-center_temp):224, 0:224] = gaussiand_image_bottom
    else:
        center_temp = (224 - resized.shape[1])//2
        if resized.shape[1]%2:
             resized_image[0:224, center_temp+1:224-center_temp] = resized[0:224, 0:resized.shape[1]]
        else:
             resized_image[0:224, center_temp:224-center_temp] = resized[0:224, 0:resized.shape[1]]
        #Fill out blank part of image
        gaussiand_image_left = cv2.GaussianBlur(resized[0:224, 0:center_temp],(7,7),1000)
        gaussiand_image_right = cv2.GaussianBlur(resized[0:224,(resized.shape[1]-center_temp):resized.shape[1]],(7,7),1000)
        resized_image[0:224, 0:center_temp] = gaussiand_image_left
        resized_image[0:224, (224-center_temp):224] = gaussiand_image_right
    
    return resized_image

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
def generateImageSet(dataframe, index=None, start=None, end=None, path=None):
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
        resized_image = changeImageShape(impath)
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
args = parser.parse_args()
matrix_path = os.path.join(args.dpath, 'imageset.npy')
image_index = os.path.join(args.dpath, 'indexinfo.csv')
front_csv_path = os.path.join(args.dpath, "cat_front.csv")
img_path = os.path.join(args.dpath, "Image_data/data/image/")

npm, dfm = generateImageSet(pd.read_csv(front_csv_path), path=img_path)

np.save(matrix_path, np.transpose(npm, (0, 3, 1, 2)))
dfm.to_csv(image_index, index=None)


