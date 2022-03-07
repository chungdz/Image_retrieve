
# Image retrieve system

## Create folders

Two folders are needed, one for storing all data files, the other one for storing model parameters.

In this document, we use *~/ir/* as example path for the first one and *~/ir/para/* as example path for the second one.

## Data preparation

Download dataset from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

Unzip all files, you can get two folders: *data* for trainset images and *sv_data* for testset images. Create foldler *~/ir/Image_data*, move two folders *data* and *sv_data* into this folder.

## Data prepocess
At the root directory of the code, run the following instructions to generate dataset for training and testing:

	python -m compcars.data_info --dpath=~/ir/
	python -m process_data.image_matrix --dpath=~/ir/ --image_info=cat_front.csv --image_root_path=Image_data/data/image/
	python -m process_data.make_train_valid --dpath=~/ir/
	python -m compcars.test_set_generation --dpath=~/ir/

To see the meaning of the arguments of Python files, for example, the meaning of *data_info.py*, use this instruction:

	python process_data/data_info.py -h

To train this model, at least one GPU is needed. Also Pytorch version bigger than 1.0 is needed. We use RTX 3090 to train the model. If the memory of GPU is not enough, try smaller backbone (e.g. ResNet50 to ResNet18) and smaller batchsize.

	python train.py --dpath=~/ir/ --save_path=~/ir/para/ --batch_size=64 --epoch=2 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50

