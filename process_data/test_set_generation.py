import pandas as pd
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm 
import json

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
    if (resized.shape[0]+1)<224:
        center_temp = (224 - resized.shape[0])//2
        
        if resized.shape[0]%2:
            resized_image[center_temp+1:224-center_temp, 0:224] = resized[0:resized.shape[0], 0:224]
        else:
            resized_image[center_temp:224-center_temp, 0:224] = resized[0:resized.shape[0], 0:224]
        
        if center_temp > resized.shape[0]:
            return np.reshape(resized_image, (3,224,224))
        #Fill out blank part of image
        gaussiand_image_top = cv2.GaussianBlur(resized[0:center_temp, 0:224],(7,7),1000)
        gaussiand_image_bottom = cv2.GaussianBlur(resized[(resized.shape[0]-center_temp):resized.shape[0], 0:224],(7,7),1000)
        resized_image[0:center_temp, 0:224] = gaussiand_image_top
        resized_image[(224-center_temp):224, 0:224] = gaussiand_image_bottom
    elif (resized.shape[1]+1)<224:
        center_temp = (224 - resized.shape[1])//2
       
        if resized.shape[1]%2:
             resized_image[0:224, center_temp+1:224-center_temp] = resized[0:224, 0:resized.shape[1]]
        else:
             resized_image[0:224, center_temp:224-center_temp] = resized[0:224, 0:resized.shape[1]]
        
        if center_temp > resized.shape[1]:
            return np.reshape(resized_image, (3,224,224))     
        #Fill out blank part of image
        gaussiand_image_left = cv2.GaussianBlur(resized[0:224, 0:center_temp],(7,7),1000)
        gaussiand_image_right = cv2.GaussianBlur(resized[0:224,(resized.shape[1]-center_temp):resized.shape[1]],(7,7),1000)
        resized_image[0:224, 0:center_temp] = gaussiand_image_left
        resized_image[0:224, (224-center_temp):224] = gaussiand_image_right
    else:
        resized_image[0:resized.shape[0], 0:resized.shape[1]] = resized[0:resized.shape[0], 0:resized.shape[1]]

    return np.reshape(resized_image, (3,224,224))





#==================================================================================#
parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
args = parser.parse_args()
path = os.path.join(args.dpath, "Image_data/sv_data/image/")  # image file path
print("file path", path)
files = os.listdir(path)

dfpath = os.path.join(args.dpath, "test_model.csv")
indexpath = os.path.join(args.dpath, "indexinfo.csv")
test_image_path = os.path.join(args.dpath, "test_image.npy")
test_set_path = os.path.join(args.dpath, "test.npy")
dictionary_path = os.path.join(args.dpath, "model_num.json")

df = pd.read_csv(dfpath, header = None)
indexset = pd.read_csv(indexpath)


image_set = []
test_set = []
index_dict = {} 
index = 0   #index for test set
for l in tqdm(files, total=len(files)):
    l1 = l + "/"
    files1 = None
    files1 = os.listdir(path + l1) 
    for images in files1:
        index += 1
        path_temp = path + l1 + images
        
        #Resize_testset_image
        
        resized_image = changeImageShape(path_temp)
        if resized_image.sum() == 0:
            exit()
        
        #find correspond carmodel number
        carmodel_number = df.iloc[int(l)-1][2]

        #find all index numbers for this model and add them all to dict
        temp = indexset.loc[indexset['Carmodel'] == carmodel_number]
        temp_array = temp['Index'].to_numpy()
        index_dict[int(l)-1] = temp_array


        #add image into matrix and testset
        image_set.append(resized_image)
        temp_set = []
        temp_set.append(index)
        temp_set.append(int(l)-1)
        test_set.append(np.array(temp_set))

np.save(test_image_path, np.array(image_set, dtype = np.uint8))
np.save(test_set_path, np.array(test_set))
saved_index = {k: v.tolist() for k, v in index_dict.items()}
json.dump(saved_index, open(dictionary_path, 'w'))
print(np.array(image_set, dtype = np.uint8).shape)
print(np.array(test_set).shape)
