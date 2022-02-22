# download from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ into data
cd xxx
mkdir data
cd data

cd ..
python -m process_data.data_info --dpath=data
python -m process_data.image_matrix --dpath=data
python -m process_data.make_train_valid --dpath=data

python train.py --dpath=/mnt/ir/ --save_path=/mnt/para/


