
# Image retrieve system

## Create folders

Two folders are needed, one for storing all data files, the other one for storing model parameters.

In this document, we use *~/cifar100/* as example path for the first one and *~/cifar100/para/* as example path for the second one.

## Data preparation

Download dataset from https://www.cs.toronto.edu/~kriz/cifar.html to get cifar-100-python.tar.gz

	tar -zxvf cifar-100-python.tar.gz
	mkdir ~/cifar100 ~/cifar100/para
	mv meta train test ~/cifar100/

## Data prepocess
Make sure that there are at least 32GB memory.

At the root directory of the code, run the following instructions to generate dataset for training and testing:

	python -m cifar.make_data --dpath=~/cifar100
	python -m process_data.make_train_valid --dpath=~/cifar100 --iname=train_info.csv --skip=4 --ratio=25

To see the meaning of the arguments of Python files, for example, the meaning of *data_info.py*, use this instruction:

	python process_data/data_info.py -h

## Training
The structure of the model is refer to [GeM](https://ieeexplore.ieee.org/abstract/document/8382272).

To train this model, at least one GPU is needed. Also Pytorch version bigger than 1.0 is needed. We use RTX 3090 to train the model. If the memory of GPU is not enough, try smaller backbone (e.g. ResNet50 to ResNet18) and smaller batchsize.

	python train_class.py --start_epoch=-1 --dpath=~/cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=~/cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin --encoder=gem

## Encoding
After training, the images in training matrix are encoded as database, and images in test matrix are encoded as query. The instructions below use the parameters saved after first epoch.

	python build_image_db.py --dpath=~/cifar100 --img_size=224 --save_path=~/cifar100/para/model.ep8 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=swin --encoder=gem
	python build_image_db.py --dpath=~/cifar100 --img_size=224 --save_path=~/cifar100/para/model.ep8 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=swin --encoder=gem
## Evaluation
mAP is used to evaluate the performance for 10000 query in testset.
	
	python -m process_data.mips_cifar

## Visualization
Check *dataset_diff_cifar.ipynb* for visualization. Given a index in image matrix and the type of the image (in training matrix or test matrix), the function returns top k images using MIPS.


