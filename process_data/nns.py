from lopq import LOPQModel, LOPQSearcher
import os
import numpy as np
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--dimension", default=64, type=int)
parser.add_argument("--k", default=20, type=int)
args = parser.parse_args()

dbp = os.path.join(args.dpath, "database.npy")
imagep = os.path.join(args.dpath, "tdatabase.npy")
classp = os.path.join(args.dpath, "model_num.json")
resp = os.path.join(args.dpath, "mAP.json")
testp = os.path.join(args.dpath, "test.npy")
print('load data')
db = np.load(dbp)
testdb = np.load(imagep)
md = json.load(open(classp, "r"))
testset = np.load(testp)
# Define a model and fit it to data
print('fit')
model = LOPQModel(V=8, M=args.dimension, subquantizer_clusters=256)
model.fit(db, verbose=True, random_state=7)

# Compute the LOPQ codes for a vector
# code = model.predict(x)

# Create a searcher to index data with the model
print('load searcher')
searcher = LOPQSearcher(model)
searcher.add_data(db)

# Retrieve ranked nearest neighbors
res = []
for imgidx, imgclass in tqdm(testset):
    results, visited = searcher.search(testdb[imgidx], quota=args.k)
    nns = [r.id for r in list(results)]
    res.append(nns)

final_matrix = np.array(res)
print(final_matrix.shape)

test_class = testset[:, 1]
labeled = np.zeros(final_matrix.shape)
mAP_list = []
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

print(sum(mAP_list) / len(mAP_list))
json.dump(mAP_list, open(resp, "w"))
