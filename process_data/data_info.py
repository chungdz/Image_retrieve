import pandas as pd
import argparse
import os
from tqdm import tqdm 

#============================================================================#
#GetImage(Path)
#Input: the location of the image set: E.g: "F:/Image_data/data/image/"
#Output: Two Dataframes: df_front_all, df_rear_all
#
#This function will generate two files with all images path and image names
#
#File1: Car_Front.csv        --->All image info with front views
#File2: Car_Rear.csv         --->All image info with rear views
#
#
#Example File:
#                                                   Path  FileName       View Point Year CarMake CarModel
#F:/Image_data/data/image/1/1101/2011/07b90decb92ba6.jpg  07b90decb92ba6 1          2011 1       1101
#F:/Image_data/data/image/1/1101/2011/7a6282504fdd2c.jpg  7a6282504fdd2c 1          2011 1       1101
#F:/Image_data/data/image/1/1101/2011/a476ea5838ce2b.jpg  a476ea5838ce2b 1          2011 1       1101
#F:/Image_data/data/image/1/1101/2011/adb149361578ad.jpg  adb149361578ad 1          2011 1       1101
#F:/Image_data/data/image/1/1102/2011/12226a5418fcff.jpg  12226a5418fcff 1          2011 1       1102
#============================================================================#

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
args = parser.parse_args()
path = os.path.join(args.dpath, "Image_data/data/image/")  # image file path
label_path = os.path.join(args.dpath, "Image_data/data/label/") # label file path
front_csv_path = os.path.join(args.dpath, "cat_front.csv")
rear_csv_path = os.path.join(args.dpath, "cat_rear.csv")
all_csv_path = os.path.join(args.dpath, "car_all.csv")
print("file path", path)
files = os.listdir(path)

viewpoints = []
carmodel_list = []
year_list = []
car_makenames = []
image_origin = []
filepath = []
filenames = []

for l in tqdm(files, total=len(files)):
    l = l + "/"
    files1 = None
    files1 = os.listdir(path + l) 
    for k in files1:
        k = k + "/"
        files2 = None
        files2 = os.listdir(path + l + k) 

        for j in files2:
            j = j + "/"
            files3 = None
            files3  = os.listdir(path + l + k + j) 

            for i in files3:
                filename, _ = i.split(".")
                content = open(label_path + l + k + j + filename + '.txt').read()
                if (content[0] != "3") & (content[0] != "-"):
                    filenames.append(filename)
                    viewpoints.append(content[0])
                    year,_ = j.split("/")
                    year_list.append(year)
                    car_make,_ = l.split("/")
                    car_makenames.append(car_make)
                    car_model,_ = k.split("/")
                    carmodel_list.append(car_model)
                    
                    filepath.append(l + k + j + filename + '.jpg')
                    #image_origin.append(cv2.imread(path + l + k + j + filename + '.jpg'))

data = {'Path': filepath, 'FileName': filenames, 
        'View Point': viewpoints, 'Year': year_list, 
        'CarMake': car_makenames, 'CarModel': carmodel_list}

df = pd.DataFrame(data)

df_front = df[df['View Point'] == "1"]
df_front_side = df[df['View Point'] == "4"]
df_front_all = pd.concat([df_front, df_front_side])
    
df_rear = df[df['View Point'] == "2"]
df_rear_side = df[df['View Point'] == "5"]
df_rear_all = pd.concat([df_rear, df_rear_side])

df_all = pd.concat([df_front_all, df_rear_all])
  
df_front_all.reset_index(inplace=True, drop=True)
df_rear_all.reset_index(inplace=True, drop=True)
df_all.reset_index(inplace=True, drop=True)

pd.set_option('display.width', None)
#print(df_front_all.head(5))
#print(df_rear_all.head(5))
print(df_all.head(5))

#df_front_all.to_csv(front_csv_path, index=None)
#df_rear_all.to_csv(rear_csv_path, index=None)
df.to_csv(all_csv_path, index=None)

