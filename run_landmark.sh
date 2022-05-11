python -m revisitop.make_test_files --dname=oxford5k --dpath=landmark
python -m revisitop.make_test_files --dname=paris6k --dpath=landmark

# Swin GeM 224 
# oxford
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/model.ep3 --batch_size=256 --input=oxford5k_dbm.npy --output=ox_database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/model.ep3 --batch_size=256 --input=oxford5k_qm.npy --output=ox_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=ox_database.npy --test_matrix=ox_tdatabase.npy --info_dict=oxford5k_info.json
# paris
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/model.ep3 --batch_size=256 --input=paris6k_dbm.npy --output=pa_database.npy --arch=swin --encoder=gem
python build_image_db.py --dpath=landmark --img_size=224 --save_path=cifar100/para/model.ep3 --batch_size=256 --input=paris6k_qm.npy --output=pa_tdatabase.npy --arch=swin --encoder=gem
python -m revisitop.mips --db_matrix=pa_database.npy --test_matrix=pa_tdatabase.npy --info_dict=paris6k_info.json

