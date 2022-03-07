
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
The structure of the model is refer to [GeM](https://ieeexplore.ieee.org/abstract/document/8382272).

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

Or [local optimized product quantization](https://openaccess.thecvf.com/content_cvpr_2014/html/Kalantidis_Locally_Optimized_Product_2014_CVPR_paper.html) (LOPQ), an approximate nearest neighbor search method, can be applied for both:

	python -m process_data.nns --dpath=ir --dimension=32 --k=20 --isValid=1 --to_test=valid_for_test.npy --test_matrix=database.npy
	python -m process_data.nns --dpath=ir --dimension=32 --k=20

## Visualization
Check *dataset_diff.ipynb* for visualization. Given a index in image matrix and the type of the image (in training matrix or test matrix), the function returns top k images using MIPS.

## Independent dataset
Note that since all images will be loaded into memory, the size of the dataset either for training or testing should not exceeds the size of the memory. For 67000 3\*224\*224 RGB images, the size of the matrix in format of uint8 numpy is 9GB. 

If new dataset is used. Then 3 files need to be generated: "indexinfo.csv" contains the column "Path", "Index", "Class" for all images in training set. "tindexinfo.csv" contains the column "Path", "Index", "Class" for all images in test set. "test.npy" contrains the index and class of test images, in shape [size, 2].

"Path" column is used to fetch images in the folders. The format is *dpath + image_root_path + p*, where *dpath* is the root path of all data (e.g. *~/ir/*), *image_root_path* is the root path of corresponding images (e.g. *Image_data/data/image/*). And *p* is the value in the "Path" column. 

"Index" is used to index images, should be in range [0, size - 1]. 

"Class' indicates the class of the images, note that in this project "Class" should be a integer.

Similar instructions needs to be run, assuming that *dpath=~/ir/*, *image_root_path=traini* for training images, and *image_root_path=testi* for test images:
	
	python -m process_data.image_matrix --dpath=~/ir/ --image_info=indexinfo.csv --image_root_path=traini --mname=imageset.npy
	python -m process_data.image_matrix --dpath=~/ir/ --image_info=tindexinfo.csv --image_root_path=testi --mname=test_image.npy
	python -m process_data.make_train_valid --dpath=ir --iname=indexinfo.csv

For training and encoding and evaluation, instructions are the same.

	

