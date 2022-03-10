import numpy as np
import json
import argparse
from torch.utils.data import DataLoader
from datasets.dl import GeMData
import torch
import os
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="ir", type=str,
                        help="root path of all data")
parser.add_argument("--batch_size", default=1024, type=int, help="searching batch size used in Pytorch DataLoader")
parser.add_argument("--k", default=20, type=int, help="top k images to return")
parser.add_argument("--to_test", default="test_masked.npy", type=str, help="query dataset")
parser.add_argument("--test_matrix", default="tdatabase.npy", type=str, help="encoded query vectors")
parser.add_argument("--isValid", default=0, type=int, help='is the encoded query matrix same as database')
args = parser.parse_args()

dbp = os.path.join(args.dpath, "database.npy")
testp = os.path.join(args.dpath, args.to_test)
imagep = os.path.join(args.dpath, args.test_matrix)
classp = os.path.join(args.dpath, "model_num.json")
resp = os.path.join(args.dpath, "mAP.json")
mask2p = os.path.join(args.dpath, "mask2.npy")
print('load data')
md = json.load(open(classp, "r"))
m2 = torch.LongTensor(np.load(mask2p)).to(0)
image_matrix = torch.FloatTensor(np.load(imagep))
testset = np.load(testp)
testinput = testset[:, 0]
test_class = testset[:, 1]
dataset = GeMData(image_matrix, torch.LongTensor(testinput))
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
db_tensor = torch.FloatTensor(np.load(dbp).T).to(0)

if args.isValid == 1:
    idxset = set(testinput)
    mask = np.ones((1, image_matrix.size(0)))
    for i in range(image_matrix.size(0)):
        if i in idxset:
            mask[0, i] = 0
    mask = torch.LongTensor(mask).to(0)

batch_res = []
with torch.no_grad():
    for data in tqdm(data_loader, total=len(data_loader), desc="generate vectors"):
        input_data = data.to(0)
        if args.isValid:
            cur_score = torch.matmul(input_data, db_tensor) * mask * m2
        else:
            cur_score = torch.matmul(input_data, db_tensor) * m2
        _, topk = torch.topk(cur_score, args.k, dim=1)
        batch_res.append(topk.cpu().numpy())

final_matrix = np.concatenate(batch_res, axis=0)
print(final_matrix.shape)

labeled = np.zeros(final_matrix.shape)
mAP_list = []
sum_list = []
for i in tqdm(range(final_matrix.shape[0]), desc='map to binary and calculate mAP'):
    cur_s = set(md[str(test_class[i])])
    plist = []
    for j in range(final_matrix.shape[1]):
        labeled[i, j] = int(final_matrix[i][j] in cur_s)
        if labeled[i, j] == 1:
            plist.append(labeled[i, :j + 1].sum() / (j + 1))

    if len(plist) != 0:
        mAP_list.append(sum(plist) / len(plist))
    else:
        mAP_list.append(0)
    sum_list.append(sum(plist))

print(sum(mAP_list) / len(mAP_list))
json.dump(mAP_list, open(resp, "w"))

top_indices = np.argsort(sum_list)[-20:]
top_pic_index = [testinput[x] for x in top_indices]
print("highest recall", top_pic_index)

common_indices = np.argsort(mAP_list)[-20:]
common_pic_index = [testinput[x] for x in common_indices]
print("highest mAP", common_pic_index)





