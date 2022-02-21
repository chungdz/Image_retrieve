import imp
import random
from utils.train_util import set_seed
from random import randrange
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import trange

set_seed(7)

#============================================================================#
# get a random index of a positive sample for trainning set and validation set 
#============================================================================#
def getrandom_pos(df, index):
    
    #get all index for this car model, random one != current one
    df_temp = df.loc[df['Index'] == index]
    
    carmodel = df_temp.iloc[0]['Carmodel']
    temp = df.loc[df['Carmodel'] == carmodel]
    random_array = temp['Index'].to_numpy()
    random_array = random_array[random_array != index]
    result = random.choice(random_array)
    
    return result


#============================================================================#
#get a random index of a negative sample for trainning set and validation set 
#============================================================================#
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
    
    
#============================================================================#
#generate train and validation sets
#============================================================================#
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
    for index in trange(132, 2, -1):
        #print(df.groupby('Carmodel').nth(index))
        df_temp = df.groupby('Carmodel').nth(index)
    
        M = df_temp['Index'].to_numpy()
        # print(M.shape)
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

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
args = parser.parse_args()
image_index = os.path.join(args.dpath, 'indexinfo.csv')
trainp = os.path.join(args.dpath, 'train.csv')
validp = os.path.join(args.dpath, 'valid.csv')

train, valid = generateTrainset(image_index)
np.save(trainp, train)
np.save(validp, valid)


