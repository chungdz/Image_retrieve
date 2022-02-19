#imports
import os
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt # PyPlot for visualize images
import numpy as np              # numpy library
import cv2                      #Open-CV libraries
import pandas as pd
from random import randrange
import random

def init(): 
    
    path = "F:/Image_data/data/image/" # image file path
    label_path = "F:/Image_data/data/label/" # label file path
    

    return path, label_path




def changeImageShape(path):
    image = None
    image  = cv2.imread(path)    
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

def Plot_image(images):
    for i in range(len(images)):
        plt.subplot(1, 10, i+1)
        plt.imshow(np.reshape(images[i], (224,224,3)))
    plt.show()


#============================================================================#
#GetImage(Path)
#Input: the location of the image set: E.g: "F:/Image_data/data/image/"
#Output: Two Dataframes: df_front_all, df_rear_all
#
#This function will generate two files with all images path and image names
#
#File1: Car_Front.txt        --->All image info with front views
#File2: Car_Rear.txt         --->All image info with rear views
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
def generateImageInfoFile(path, label_path):
    files = os.listdir(path)

    viewpoints = []
    carmodel_list = []
    year_list = []
    car_makenames = []
    image_origin = []
    filepath = []
    filenames = []

    for l in files:
        l = l + "/"
        files1 = os.listdir(path + l) 
        for k in files1:
            k = k + "/"
            files2 = os.listdir(path + l + k) 
    
            for j in files2:
                j = j + "/"
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
                        
                        filepath.append(path + l + k + j + filename + '.jpg')
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
        
    df_front_all.reset_index(inplace=True, drop=True)
    df_rear_all.reset_index(inplace=True, drop=True)
    
    pd.set_option('display.width', None)
    print(df_front_all.head(5))
    print(df_rear_all.head(5))

    df_front_all.to_csv(r'C:\Users\asus-pc\Desktop\COEN340_Computer_Vision\Final Project\Car_Front.csv')
    df_rear_all.to_csv(r'C:\Users\asus-pc\Desktop\COEN340_Computer_Vision\Final Project\Car_Rear.csv')
    #print(len(image_origin))

    return df_front_all, df_rear_all





def generateImageSet(dataframe, index, start, end):
    
    #generateImageSet(df_front_all)
    
    image_set = []
    index_set = []
    model_set = []
    #get image one by one:
    
    for image_num in range(start, end):
        #generate 10 sub image for one image
        resized_image = changeImageShape(dataframe["Path"].iloc[image_num])

        image_set.append(resized_image)
        index_set.append(index)
        model_set.append(dataframe["CarModel"].iloc[image_num])
        index+=1
        print("{}: Completed".format(index))
    imageset = np.array(image_set, dtype = np.uint8)

    #save index info as a dataframe
    data = {'Index': index_set, 'Carmodel': model_set}
    
    index_df = pd.DataFrame(data)
    return imageset, index_df
    
#Save Imageset and Index Dataframe:
def savetofile(imagepath, indexpath, imageset, indexset):
    
    np.save(imagepath, np.array(imageset, dtype = np.uint8))
    indexset.to_csv(indexpath, index = False)

    

def getrandom_pos(df, index):
    
    #get all index for this car model, random one != current one
    df_temp = df.loc[df['Index'] == index]
    
    carmodel = df_temp.iloc[0]['Carmodel']
    temp = df.loc[df['Carmodel'] == carmodel]
    random_array = temp['Index'].to_numpy()
    random_array = random_array[random_array != index]
    result = random.choice(random_array)
    
    return result


def getrandom_neg(df, index):
    
    df_temp = df.loc[df['Index'] == index]
    carmodel = df_temp.iloc[0]['Carmodel']
    temp = df.Carmodel.unique()
    
    rand_num = random.choice(temp)
    while rand_num == carmodel:   #in case choose the same model
        rand_num = rand_num = random.choice(temp)
    
    temp = df.loc[df['Carmodel'] == rand_num]
    random_array = temp['Index'].to_numpy()
    result = random.choice(random_array)
    
    return result
    
    
def generateTrainset(path):
    df = pd.read_csv(path)

    trainset = []
    validset = []

    for i in range(1, 3):
    #add all last two images to validate set
        df_temp = df.groupby('Carmodel').nth(i)
    
        X = df_temp['Index'].to_numpy()
        #print(M.shape)
        for x in X:
            templist = []
            templist.append(x)
            templist.append(getrandom_pos(df, x))
            templist.append(getrandom_neg(df, x))
            validset.append(np.array(templist))






    count = 0
    for index in range(132, 2, -1):
        #print(df.groupby('Carmodel').nth(index))
        df_temp = df.groupby('Carmodel').nth(index)
    
        M = df_temp['Index'].to_numpy()
        print(M.shape)
        for m in M:
            if count == 2:
                templist = []
                templist.append(m)
                #append one positive and 3 negative
                templist.append(getrandom_pos(df, m))
                templist.append(getrandom_neg(df, m))
                validset.append(np.array(templist))
            elif count == 4:
                templist = []
                templist.append(m)
                #append one positive and 3 negative
                templist.append(getrandom_pos(df, m))
                templist.append(getrandom_neg(df, m))
                validset.append(np.array(templist))
            else:
                templist = []
                templist.append(m)
                #append one positive and 3 negative
                templist.append(getrandom_pos(df, m))
                templist.append(getrandom_neg(df, m))
                templist.append(getrandom_neg(df, m))
                templist.append(getrandom_neg(df, m))
                templist.append(getrandom_neg(df, m))
                trainset.append(np.array(templist))
    
        count+=1
    
        if count == 6:
            count = 0
    
                
    
    trainset_res = np.array(trainset)
    validset_res = np.array(validset)
    print(np.array(trainset).shape)
    print(np.array(validset).shape)
    return trainset_res, validset_res












def main():
    #path, label_path = init()
    
    df_front_all = pd.read_csv(r'C:\Users\asus-pc\Desktop\COEN340_Computer_Vision\Final Project\Car_Front.csv')
    df_rear_all = pd.read_csv(r'C:\Users\asus-pc\Desktop\COEN340_Computer_Vision\Final Project\Car_rear.csv')
    
    print(df_front_all.shape)

    #Change with your won data path
    path1  = 'F:\\Data\\imageset'           #matrix with all images
    path2 = 'F:\Data\indexinfo.csv'         #index information dataframe
    #imageset, index_df = generateImageSet(df_front_all, 0, 0, 67732)
    #savetofile(path1, path2, imageset, index_df)
    trainset, validset = generateTrainset(path2)
    np.save('F:\\Data\\trainset' ,trainset)
    np.save('F:\\Data\\validset' ,validset)
    
    
    

if __name__ == "__main__":
    main()

