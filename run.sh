# download from http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/ into data
cd xxx
mkdir data
cd data

cd ..
python -m process_data.data_info
python -m process_data.image_matrix
python -m process_data.make_train_valid.py
