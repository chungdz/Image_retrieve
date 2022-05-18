python -m revisitop.make_test_files --dname=oxford5k --dpath=landmark
python -m revisitop.make_test_files --dname=paris6k --dpath=landmark
python -m revisitop.make_train_info
# whole
python -m revisitop.merge
python -m process_data.make_train_valid --dpath=landmark --ratio=16 --iname=train_info_final.csv --skip=10
# only revisit
python -m revisitop.merge_no_sfm
python -m process_data.make_train_valid --dpath=landmark --ratio=16 --iname=revisit_info.csv --skip=10

# using --isM=0 for single gem
# Swin GeM 224 
python train_class.py --isM=0 --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=8 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin
# oxford
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep4 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=swin --encoder=gem
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep4 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=swin --encoder=gem
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json
# Swin Multi GeM 224 
python train_class.py --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin
# python train_class.py --start_epoch=-1 --dpath=landmark --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=10 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=swin --mfile=revisit_trainset.npy --train_num=1
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep8 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep8 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep8 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep8 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json

# Resnet 50 + GEM 224 x 224
python train_class.py --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=6 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=resnet50
python build_image_db.py  --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=resnet50
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=resnet50
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=resnet50
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json
# ResNet 50 Single
python train_class.py --isM=0 --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=5 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet50
# oxford
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep3 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=resnet50
python build_image_db.py  --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep3 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=resnet50
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep3 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=resnet50
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep3 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=resnet50
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json
# Resnet 101 + GEM 224 x 224
# single gem
python train_class.py --isM=0 --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=5 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet101
# oxford
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep4 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=resnet101
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep4 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=resnet101
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep4 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=resnet101
python build_image_db.py --isM=0 --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep4 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=resnet101
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json

# multi gem
python train_class.py --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=6 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=resnet101
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=resnet101
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=resnet101
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=resnet101
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=resnet101
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json

# DeiT
python train_class.py --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=6 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=deit
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep7 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=deit
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep7 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=deit
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep7 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=deit
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep7 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=deit
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json

# DeiT Single Gem
python train_class.py --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=7 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=deitgem
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=deitgem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=deitgem
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=deitgem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep6 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=deitgem
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json

# DeiT multi Gem
python train_class.py --start_epoch=-1 --dpath=landmark --train_num=6 --img_size=224 --save_path=landmark/para/ --batch_size=64 --epoch=7 --show_batch=5 --lr=0.0001 --lr_shrink=0.9 --arch=deitmulti
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=deitmulti
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=deitmulti
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=deitmulti
python build_image_db.py --dpath=landmark --img_size=224 --save_path=landmark/para/model.ep2 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=deitmulti
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json
