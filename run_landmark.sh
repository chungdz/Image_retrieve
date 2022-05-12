python -m revisitop.make_test_files --dname=oxford5k --dpath=landmark
python -m revisitop.make_test_files --dname=paris6k --dpath=landmark
python -m revisitop.make_train_info
python -m process_data.make_train_valid --dpath=landmark --ratio=16 --iname=train_info.csv --skip=10

# Swin GeM 224 
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/weighted_sum.layer_norm --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/weighted_sum.layer_norm --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/weighted_sum.layer_norm --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/weighted_sum.layer_norm --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json


# Resnet 101 + GEM 224 x 224
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/resnet101.multigem --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=resnet101
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/resnet101.multigem --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=resnet101
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/resnet101.multigem --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=resnet101
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/resnet101.multigem --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=resnet101
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json
