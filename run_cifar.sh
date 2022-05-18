python -m cifar.make_data --dpath=cifar100 --img_size=224
python -m process_data.make_train_valid --dpath=cifar100 --iname=train_info.csv --skip=4 --ratio=25
# use --isM=0 to change multi gem to single gem 
# Resnet 50 + GEM 224 x 224
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=20 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep14 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=resnet50
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep14 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=resnet50
# Resnet 101 + GEM 224 x 224
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=20 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet101
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep15 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=resnet101
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep15 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=resnet101
# Swin + ATT 224 224 
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin --encoder=att
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep1 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=swin --encoder=att
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep1 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=swin --encoder=att
# Swin + GEM 224 224 
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep3 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep3 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=swin --encoder=gem
# DeiT + CLS 244 244
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=deit --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep6 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=deit --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep6 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=deit --encoder=gem
# DeiT + Single GeM 244 244
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=deitgem --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep5 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=deitgem --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep5 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=deitgem --encoder=gem
# DeiT multi gem 224 224
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=deitmulti --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep6 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=deitmulti --encoder=gem
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep6 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=deitmulti --encoder=gem
# MViT + CLS 244 244
python train_class.py --start_epoch=-1 --dpath=cifar100 --mfile=train_image_set.npy --img_size=224 --save_path=cifar100/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.001 --lr_shrink=0.9 --arch=mvit
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep6 --batch_size=256 --input=train_image_set.npy --output=database.npy --arch=mvit
python build_image_db.py --dpath=cifar100 --img_size=224 --save_path=cifar100/para/model.ep6 --batch_size=256 --input=test_image_set.npy --output=tdatabase.npy --arch=mvit
# evaluate
python -m process_data.mips_cifar



