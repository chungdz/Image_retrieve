import numpy as np
import argparse
import os
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
args = parser.parse_args()

testsetpath = os.path.join(args.dpath, "test.npy")
testset_maskedpath = os.path.join(args.dpath, "test_masked.npy")
discp = os.path.join(args.dpath, "cutted_class.json")
dbp = os.path.join(args.dpath, "database.npy")
dbp_masked = os.path.join(args.dpath, "database_masked.npy")
indexpath = os.path.join(args.dpath, "indexinfo.csv")
mask2p = os.path.join(args.dpath, "mask2.npy")
dbcmp = os.path.join(args.dpath, "dbcm.npy")
print('load data')
fdf = pd.read_csv(indexpath)
pic2cm = {}
for pic_index, carm in fdf.values.tolist():
    pic2cm[pic_index] = carm
testset = np.load(testsetpath)
discarded = set(json.load(open(discp, "r")))
db = np.load(dbp)

testset_masked = []
for pic_idx, cmclass in testset:
    if int(cmclass) in discarded:
        continue
    testset_masked.append([pic_idx, cmclass])
testset_masked = np.array(testset_masked)
print("masked test set shape", testset_masked.shape)

mask2 = []
dbcm = []
db_masked = []
for i in range(db.shape[0]):
    if pic2cm[i] in discarded: 
        mask2.append(0)
    else:
        mask2.append(1)
        db_masked.append(db[i].tolist())
        dbcm.append(pic2cm[i])
mask2 = np.array(mask2).reshape(1, -1)
db_masked = np.array(db_masked)
dbcm = np.array(dbcm)
print("masked 2 shape", mask2.shape, "trimmed database shape", db_masked.shape)

print('save data')
np.save(testset_maskedpath, testset_masked)
np.save(mask2p, mask2)
np.save(dbp_masked, db_masked)
np.save(dbcmp, dbcm)




