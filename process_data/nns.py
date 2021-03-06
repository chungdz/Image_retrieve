from lopq import LOPQModel, LOPQSearcher
import os
import numpy as np
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="root path of all data")
parser.add_argument("--dimension", default=64, type=int, help='''number of fine codes; 
                                same as number of bytes per compressed vector in memory with 
                                256 subquantizer clusters''')
parser.add_argument("--k", default=20, type=int, help="top k images to return")
parser.add_argument("--to_test", default="test_masked.npy", type=str, help="query dataset")
parser.add_argument("--test_matrix", default="tdatabase.npy", type=str, help="encoded query vectors", help="encoded query vectors")
parser.add_argument("--isValid", default=0, type=int, help='is the encoded query matrix same as database')
args = parser.parse_args()

dbp = os.path.join(args.dpath, "database.npy")
dbp_masked = os.path.join(args.dpath, "database_masked.npy")
# classp = os.path.join(args.dpath, "model_num.json")
resp = os.path.join(args.dpath, "mAP.json")
testp = os.path.join(args.dpath, args.to_test)
imagep = os.path.join(args.dpath, args.test_matrix)
dbcmp = os.path.join(args.dpath, "dbcm.npy")
print('load data')
db = np.load(dbp)
db_masked = np.load(dbp_masked)
dbcm = np.load(dbcmp)
testdb = np.load(imagep)
# md = json.load(open(classp, "r"))
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
searcher.add_data(db_masked)

# Retrieve ranked nearest neighbors
res = []
for imgidx, imgclass in tqdm(testset):
    if args.isValid:
        results, visited = searcher.search(testdb[imgidx], quota=args.k + 1)
        nns = [r.id for r in list(results)][1:]
    else:
        results, visited = searcher.search(testdb[imgidx], quota=args.k)
        nns = [r.id for r in list(results)]
    res.append(nns)

final_matrix = np.array(res)
print(final_matrix.shape)

test_class = testset[:, 1]
labeled = np.zeros(final_matrix.shape)
mAP_list = []
for i in tqdm(range(final_matrix.shape[0]), desc='map to binary and calculate mAP'):
    # cur_s = set(md[str(test_class[i])])
    plist = []
    for j in range(final_matrix.shape[1]):
        labeled[i, j] = int(dbcm[final_matrix[i][j]] == test_class[i])
        if labeled[i, j] == 1:
            plist.append(labeled[i, :j + 1].sum() / (j + 1))

    if len(plist) != 0:
        mAP_list.append(sum(plist) / len(plist))
    else:
        mAP_list.append(0)

print(sum(mAP_list) / len(mAP_list))
json.dump(mAP_list, open(resp, "w"))
