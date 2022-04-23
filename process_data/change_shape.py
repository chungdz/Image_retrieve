from re import template
from unittest.mock import NonCallableMagicMock
import cv2
import numpy as np
import argparse
import os
import pandas as pd
from tqdm import tqdm, trange


#============================================================================#
#image_filter(image, filter_size=7, sigma=1000, filter_type='Gaussian')
#Input: image: input image for filtering
#       filter_size: kernel size for filtering
#       sigma: sigma value for Gaussian Filter only
#       filter_type: what type of filter to apply.
#Output: Single image after filtering
#
#============================================================================#
def image_filter(image, filter_size=7, sigma=1000, filter_type='Gaussian'):
    if filter_type == 'Gaussian':
        result = cv2.GaussianBlur(image, (filter_size,filter_size), sigma)
    elif filter_type == "Black":
        image[:] = 0
        result = image
    elif filter_type == "White":
        image[:] = 225
        result = image
    
    return result


#============================================================================#
#largeRatioimage(image)
#
#Input: origin_image: first resized image matrix
#       
#       resized_image: res*res*channel image with origin_image image in the center
#       
#       longerSide: indicator of which side is longer (height or width)
#       
#
#Output: Single image matrix
#
#Description:
# This function is used to process images which any side of the reshaped image 
#   with length shorter than 1/3 of designed resolution (res) matrix in 
#   the changeImageShape function. 
#============================================================================#
def largeRatioimage(origin_image, resized_image, longerSide, content_start_location):
    #apply gaussian filter to the whole image:
    temp_matrix = image_filter(origin_image, filter_size=7, sigma=1000, filter_type='Gaussian')
    #detect last zero low or column number
    if resized_image[content_start_location].sum() != 0:
        content_start_location = content_start_location - 1
    
    if longerSide == "Width":
        resized_image[content_start_location-origin_image.shape[0]:content_start_location, 0:resized_image.shape[1]] = temp_matrix
        resized_image[content_start_location+origin_image.shape[0]:(content_start_location+2*origin_image.shape[0]), 0:resized_image.shape[1]] = temp_matrix
    elif longerSide == "Height":
        resized_image[0:resized_image.shape[0], content_start_location-origin_image.shape[1]:content_start_location] = temp_matrix
        resized_image[0:resized_image.shape[0], content_start_location+origin_image.shape[1]:(content_start_location+2*origin_image.shape[1])] = temp_matrix
    return resized_image
    

#============================================================================#
#changeImageShape_Gaussian(path, res=224, numChannels=3, filter_type=Gaussian,)
#Input: the location of single image: E.g: "F:/Image_data/data/image/"
#Output: Single image with shape of resXresXnumChannels
#
# Image first reshape scale to longer side = res
# Full fill empty part with origin part of image after different filters
#============================================================================#
def changeImageShape(path, res=224, numChannels=3, filter_type='Gaussian', sigma=1000, filter_size=7):
    image = None               #clear image variable in case memory use error from imread()
    image  = cv2.imread(path)    
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert BGR to RGB
    if image1.shape[1] == image1.shape[2]:
        return cv2.resize(image1, (res, res), interpolation=cv2.INTER_LINEAR)

    if image1.shape[1] < image1.shape[0]:
        scale_percent = image1.shape[0]/res
        width = round(image1.shape[1] / scale_percent)
        height = round(image1.shape[0] / scale_percent)
    else:
        scale_percent = image1.shape[1]/res
        width = round(image1.shape[1] / scale_percent)
        height = round(image1.shape[0] / scale_percent)

    dim = (width, height)
    resized = cv2.resize(image1, dim, interpolation=cv2.INTER_LINEAR)
    #create an empty array with size of res*res*numChannels:
    resized_image = np.zeros((res,res,numChannels),dtype=np.uint8)
    
    #Copy resized image into numChannels00*numChannels00*numChannels matrix
    if (resized.shape[0]+1) < res: #width greater than height
        center_temp = (res - resized.shape[0])//2
        if resized.shape[0]%2:
            resized_image[center_temp+1:res-center_temp, 0:res] = resized[0:resized.shape[0], 0:res]
        else:
            resized_image[center_temp:res-center_temp, 0:res] = resized[0:resized.shape[0], 0:res]
        

        #Fill out blank part of image
        if center_temp > resized.shape[0]:
            return largeRatioimage(resized, resized_image, longerSide="Width", content_start_location = center_temp)
        else:
            filterd_part_top = image_filter(resized[0:center_temp, 0:res], filter_size, sigma, filter_type)
            filterd_part_bottom = image_filter(resized[(resized.shape[0]-center_temp):resized.shape[0], 0:res], filter_size, sigma, filter_type)
            resized_image[0:center_temp, 0:res] = filterd_part_top
            resized_image[(res-center_temp):res, 0:res] = filterd_part_bottom
    
    elif (resized.shape[1]+1) < res:
        center_temp = (res - resized.shape[1])//2
        if resized.shape[1]%2:
             resized_image[0:res, center_temp+1:res-center_temp] = resized[0:res, 0:resized.shape[1]]
        else:
             resized_image[0:res, center_temp:res-center_temp] = resized[0:res, 0:resized.shape[1]]
        

        #Fill out blank part of image
        if center_temp > resized.shape[1]:
            return largeRatioimage(resized, resized_image, longerSide="Height", content_start_location = center_temp)
        else:
            filterd_part_left = image_filter(resized[0:res, 0:center_temp], filter_size, sigma, filter_type)
            filterd_part_right = image_filter(resized[0:res,(resized.shape[1]-center_temp):resized.shape[1]], filter_size, sigma, filter_type)
            resized_image[0:res, 0:center_temp] = filterd_part_left
            resized_image[0:res, (res-center_temp):res] = filterd_part_right
    
    else:
        resized_image[0:resized.shape[0], 0:resized.shape[1]] = resized[0:resized.shape[0], 0:resized.shape[1]]
    
        
    return resized_image