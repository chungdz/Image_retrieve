# download parameter files from https://github.com/microsoft/Swin-Transformer
# map22kto1k.txt is in ./data/
# swin_large_patch4_window7_224_22k.yaml is in ./configs/
# swin_large_patch4_window7_224_22k.pth is in https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
mkdir swin_para
# move three files into swin_para

python -m cifar.make_data --dpath=cifar100 --img_size=224
python -m process_data.make_train_valid --dpath=cifar100 --iname=train_info.csv --skip=4 --ratio=25

# Resnet 50 + GEM 224 x 224
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep9 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep9 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=resnet50
# Swin + ATT 224 224 
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin --encoder=att
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep1 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=swin --encoder=att
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep1 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=swin --encoder=att
# Swin + GEM 224 224 
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep8 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep8 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=swin --encoder=gem
# evaluate
python -m process_data.mips_cifar


