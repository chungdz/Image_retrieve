# Image retrieve system

## Create folders

Two folders are needed, one for storing all data files, the other one for storing model parameters.

In this document, we use *~/ir/* as example path for the first one and *~/ir/para/* as example path for the second one.

## Data preparation

Download dataset from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

Unzip all files, you can get two folders: *data* for trainset images and *sv_data* for testset images. Create foldler *~/ir/Image_data*, move two folders *data* and *sv_data* into this folder.

## Data prepocess
At the root directory of the code, run the following instructions to generate dataset for training and testing:

	python -m process_data.data_info --dpath=~/ir/
	python -m process_data.image_matrix --dpath=~/ir/
	python -m process_data.make_train_valid --dpath=~/ir/
	python -m process_data.test_set_generation --dpath=~/ir/

