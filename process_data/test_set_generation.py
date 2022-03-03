import pandas as pd
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm 
import json
from process_data.change_shape import changeImageShape





#==================================================================================#
parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--filter_type", default="Gaussian", type=str,
                        help="Filter type for empty space of images")
parser.add_argument("--image_resolution", default=224, type=int,
                        help="resized image resolution: e.g: 224")
parser.add_argument("--numChannels", default=3, type=int,
                        help="number of channels for images. e.g: 3")

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
        path_temp = path + l1 + images
        
        #Resize_testset_image
        
        resized_image = changeImageShape(path_temp, filter_type=args.filter_type, res=args.image_resolution, numChannels=args.numChannels, sigma=1000, filter_size=7)
        if resized_image.sum() == 0:
            exit()
        
        #find correspond carmodel number
        carmodel_number = df.iloc[int(l)-1][2]

        #find all index numbers for this model and add them all to dict
        temp = indexset.loc[indexset['Carmodel'] == carmodel_number]
        temp_array = temp['Index'].to_numpy()
        index_dict[carmodel_number] = temp_array


        #add image into matrix and testset
        image_set.append(resized_image)
        temp_set = []
        temp_set.append(index)
        temp_set.append(carmodel_number)
        test_set.append(np.array(temp_set))
        index += 1

npm = np.array(image_set, dtype = np.uint8)
np.save(test_image_path, np.transpose(npm, (0, 3, 1, 2)))
np.save(test_set_path, np.array(test_set))
# saved_index = {k: v.tolist() for k, v in index_dict.items()}
# json.dump(saved_index, open(dictionary_path, 'w'))
print(np.array(image_set, dtype = np.uint8).shape)
print(np.array(test_set).shape)
