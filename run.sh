# download from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ into data
cd xxx
mkdir data
cd data

cd ..
python -m process_data.data_info --dpath=ir
python -m process_data.image_matrix --dpath=ir
python -m process_data.make_train_valid --dpath=ir

python train.py --dpath=ir --save_path=ir/para/ --batch_size=128 --epoch=3 --show_batch=5 --lr=0.0001



