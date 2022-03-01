# download from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ into data
cd xxx
mkdir data
cd data

cd ..
python -m process_data.data_info --dpath=ir
python -m process_data.image_matrix --dpath=ir
python -m process_data.make_train_valid --dpath=ir
python -m process_data.test_set_generation --dpath=ir

python train.py --dpath=ir --save_path=ir/para/ --batch_size=64 --epoch=3 --show_batch=5 --lr=0.0001 --lr_shrink=0.9
python build_image_db.py --dpath=ir --save_path=ir/para/model.ep0 --batch_size=256 --input=imageset.npy --output=database.npy
python build_image_db.py --dpath=ir --save_path=ir/para/model.ep0 --batch_size=256 --input=test_image.npy --output=tdatabase.npy
# Maximum Inner Product Search
python mips.py --dpath=ir --batch_size=1024 --k=20
python mips.py --dpath=ir --batch_size=1024 --k=20 --to_test=valid_for_test.npy --test_matrix=database.npy
# Or LOPQ
python -m process_data.nns --dpath=ir --dimension=32 --k=20
python -m process_data.nns --dpath=ir --dimension=32 --k=20 --to_test=valid_for_test.npy --test_matrix=database.npy




