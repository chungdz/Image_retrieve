# download from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ into data
cd xxx
mkdir data
cd data

cd ..
python -m compcars.data_info --dpath=ir
python -m compcars.test_data_info --dpath=ir
python -m process_data.image_matrix --dpath=ir --image_info=car_front.csv --image_root_path=Image_data/data/image/ --mname=imageset.npy
python -m process_data.image_matrix --dpath=ir --image_info=tindexinfo.csv --image_root_path=Image_data/sv_data/image/ --mname=test_image.npy
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

python train.py --dpath=ir --save_path=ir/para/ --batch_size=64 --epoch=2 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50 --scale=0.7071 --start_epoch=0
python train.py --dpath=ir --save_path=ir/para/ --batch_size=16 --epoch=3 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50 --scale=1.4147 --start_epoch=1


python -m cifar.make_data --dpath=cifar100
python -m process_data.make_train_valid --dpath=cifar100 --iname=train_info.csv
python train.py --dpath=cifar100 mfile=train_image_set.npy --save_path=cifar100/para/ --batch_size=64 --epoch=3 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50


