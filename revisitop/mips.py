import numpy as np
import json
import argparse
from torch.utils.data import DataLoader
from datasets.dl import GeMData
import torch
import pandas as pd
import os
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="landmark", type=str,
                        help="root path of all data")
parser.add_argument("--batch_size", default=1024, type=int, help="searching batch size used in Pytorch DataLoader")
parser.add_argument("--k", default=4993, type=int, help="return all image, 4993 for oxford, 6322 for paris")
parser.add_argument("--db_matrix", default="ox_database.npy", type=str, help="else is pa_database.npy")
parser.add_argument("--test_matrix", default="ox_tdatabase.npy", type=str, help="else is pa_tdatabase.npy")
parser.add_argument("--info_dict", default="oxford5k_info.json", type=str, help="else is paris6k_info.json")
args = parser.parse_args()

dbp = os.path.join(args.dpath, args.db_matrix)
imagep = os.path.join(args.dpath, args.test_matrix)
classp = os.path.join(args.dpath, args.info_dict)
resp = os.path.join(args.dpath, "mAP.json")
print('load data')
md = json.load(open(classp, "r"))
image_matrix = torch.FloatTensor(np.load(imagep))
dataset = GeMData(image_matrix, torch.LongTensor(np.arange(70)))
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
db_tensor = torch.FloatTensor(np.load(dbp).T).to(0)

batch_res = []
with torch.no_grad():
    for data in tqdm(data_loader, total=len(data_loader), desc="generate vectors"):
        input_data = data.to(0)
        cur_score = torch.matmul(input_data, db_tensor)
        _, topk = torch.topk(cur_score, args.k, dim=1)
        batch_res.append(topk.cpu().numpy())

final_matrix = np.concatenate(batch_res, axis=0)
print(final_matrix.shape)

mAP_m_list = []
mAP_h_list = []
sum_list = []
for i in tqdm(range(final_matrix.shape[0]), desc='map to binary and calculate mAP'):
    easy = set(md[i]['easy'])
    hard = set(md[i]['hard'])
    medium_list = []
    hard_list = []
    total_m = 0
    total_h = 0
    for j in range(final_matrix.shape[1]):
        if final_matrix[i][j] in easy or final_matrix[i][j] in hard:
            total_m += 1
            medium_list.append(total_m / (j + 1))
        if final_matrix[i][j] in hard:
            total_h += 1
            hard_list.append(total_h / (j + 1))

    mAP_m_list.append(sum(medium_list) / (len(easy) + len(hard)))
    mAP_h_list.append(sum(hard_list) / len(hard))
    

print('medium', sum(mAP_m_list) / len(mAP_m_list), 'hard', sum(mAP_h_list) / len(mAP_h_list))
mdict = {
    'medium': mAP_m_list,
    'hard': mAP_h_list
}
json.dump(mdict, open(resp, "w"))

common_indices = np.argsort(mAP_m_list)[-20:]
common_pic_index = [testinput[x] for x in common_indices]
print("highest mAP for medium", common_pic_index)





