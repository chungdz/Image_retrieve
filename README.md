
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
	python -m compcars.test_data_info --dpath=~/ir/
	python -m process_data.image_matrix --dpath=~/ir/ --image_info=car_front.csv --image_root_path=Image_data/data/image/ --mname=imageset.npy
	python -m process_data.image_matrix --dpath=~/ir/ --image_info=tindexinfo.csv --image_root_path=Image_data/sv_data/image/ --mname=test_image.npy
	python -m process_data.make_train_valid --dpath=ir

To see the meaning of the arguments of Python files, for example, the meaning of *data_info.py*, use this instruction:

	python process_data/data_info.py -h

## Training
To train this model, at least one GPU is needed. Also Pytorch version bigger than 1.0 is needed. We use RTX 3090 to train the model. If the memory of GPU is not enough, try smaller backbone (e.g. ResNet50 to ResNet18) and smaller batchsize.

	python train.py --dpath=~/ir/ --save_path=~/ir/para/ --batch_size=64 --epoch=2 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50

## Encoding
After training, the images in training matrix are encoded as database, and images in test matrix are encoded as query. The instructions below use the parameters saved after first epoch.

	python build_image_db.py --dpath=~/ir/ --save_path=~/ir/para/model.ep0 --batch_size=64 --input=imageset.npy --output=database.npy --arch=resnet50
	python build_image_db.py --dpath=~/ir/ --save_path=~/ir/para/model.ep0 --batch_size=64 --input=test_image.npy --output=tdatabase.npy --arch=resnet50
## Evaluation
Then masks are generated for validation set. Because the query from validation set are in the training matrix. Same images should be masked out to avoid incorrect evaluation. And images in minor classes are masked out for better evaluation results.
	
	python -m process_data.mask_data --dpath=~/ir/

Maximum inner product search (MIPS) for both validation and test can be used for final evaluation. At least one GPU is needed.

	python -m process_data.mips --dpath=ir --batch_size=1024 --k=20 --isValid=1 --to_test=valid_for_test.npy --test_matrix=database.npy
	python -m process_data.mips --dpath=ir --batch_size=1024 --k=20

Or local optimized product quantization (LOPQ) can be applied for both:

	python -m process_data.nns --dpath=ir --dimension=32 --k=20 --isValid=1 --to_test=valid_for_test.npy --test_matrix=database.npy
	python -m process_data.nns --dpath=ir --dimension=32 --k=20

## Visualization
Check *dataset_diff.ipynb* for visualization. Given a index in image matrix and the type of the image (in training matrix or test matrix), the function returns top k images using MIPS.
