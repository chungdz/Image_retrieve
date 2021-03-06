# download from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ into data
cd xxx
mkdir data
cd data

cd ..
python -m compcars.data_info --dpath=ir
python -m compcars.test_data_info --dpath=ir
python -m compcars.unify_class_name --dpath=ir
python -m process_data.image_matrix --dpath=ir --image_info=car_front.csv --image_root_path=Image_data/data/image/ --mname=imageset.npy
python -m process_data.image_matrix --dpath=ir --image_info=tindexinfo.csv --image_root_path=Image_data/sv_data/image/ --mname=test_image.npy
# Triplet loss
python -m process_data.make_train_valid --dpath=ir

python train.py --dpath=ir --save_path=ir/para/ --batch_size=64 --epoch=3 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
python build_image_db.py --dpath=ir --save_path=ir/para/model.ep0 --batch_size=64 --input=imageset.npy --output=database.npy --arch=resnet50
python build_image_db.py --dpath=ir --save_path=ir/para/model.ep0 --batch_size=64 --input=test_image.npy --output=tdatabase.npy --arch=resnet50
python -m process_data.mask_data --dpath=ir
# Maximum Inner Product Search
python -m process_data.mips --dpath=ir --batch_size=1024 --k=20 --isValid=1 --to_test=valid_for_test.npy --test_matrix=database.npy
python -m process_data.mips --dpath=ir --batch_size=1024 --k=20
# Or LOPQ
python -m process_data.nns --dpath=ir --dimension=32 --k=20 --isValid=1 --to_test=valid_for_test.npy --test_matrix=database.npy
python -m process_data.nns --dpath=ir --dimension=32 --k=20

# Try different scale in different epoch
python train.py --dpath=ir --save_path=ir/para/ --batch_size=64 --epoch=2 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50 --scale=0.7071 --start_epoch=0
python train.py --dpath=ir --save_path=ir/para/ --batch_size=16 --epoch=3 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50 --scale=1.4147 --start_epoch=1

## classification loss
python -m process_data.make_train_valid --dpath=ir
python train_class.py --start_epoch=-1 --dpath=ir --mfile=imageset.npy --img_size=224 --save_path=ir/para/ --batch_size=64 --epoch=20 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50

python build_image_db.py --dpath=ir --save_path=ir/para/model.ep18 --batch_size=64 --input=imageset.npy --output=database.npy --arch=resnet50
python build_image_db.py --dpath=ir --save_path=ir/para/model.ep18 --batch_size=64 --input=test_image.npy --output=tdatabase.npy --arch=resnet50
python -m process_data.mask_data --dpath=ir
python -m process_data.mips --dpath=ir --batch_size=1024 --k=20

# download CIFAR-100 Python version from https://www.cs.toronto.edu/~kriz/cifar.html to get cifar-100-python.tar.gz
# decompress the file:
tar -zxvf cifar-100-python.tar.gz

# make new directory 
mkdir cifar100 cifar100/para
# mv file into directory
mv meta train test cifar100/
# come back to root and run these instructions
python -m cifar.make_data --dpath=cifar100
python -m process_data.make_train_valid --dpath=cifar100 --iname=train_info.csv --skip=4 --ratio=25

# triplet loss
python train.py --dpath=cifar100 --mfile=train_image_set.npy --img_size=64 --save_path=cifar100/para/ --batch_size=256 --epoch=1 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=64 --save_path=cifar100/para/model.ep0 --batch_size=1024 --input=train_image_set.npy --output=database.npy --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=64 --save_path=cifar100/para/model.ep0 --batch_size=1024 --input=test_image_set.npy --output=tdatabase.npy --arch=resnet50
python -m process_data.mips_cifar
# classification loss
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=64 --save_path=cifar100/para/ --batch_size=256 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=64 --save_path=cifar100/para/model.ep8 --batch_size=1024 --input=train_image_set.npy --output=database.npy --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=64 --save_path=cifar100/para/model.ep8 --batch_size=1024 --input=test_image_set.npy --output=tdatabase.npy --arch=resnet50
python -m process_data.mips_cifar
