import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm 
import json


parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="root path of all data")
args = parser.parse_args()

fp = os.path.join(args.dpath, "car_front.csv")
rp = os.path.join(args.dpath, "car_rear.csv")
tp = os.path.join(args.dpath, "tindexinfo.csv")
tnpy = os.path.join(args.dpath, "test.npy")

front_csv = pd.read_csv(fp)
rear_csv = pd.read_csv(rp)
test_csv = pd.read_csv(tp)

idict = {}
cidx = 0
for carm in front_csv['Class']:
    if carm not in idict:
        idict[carm] = cidx
        cidx += 1
print(len(idict))

front_csv['Class'] = front_csv['Class'].map(idict)
rear_csv['Class'] = rear_csv['Class'].map(idict)
test_csv['Class'] = test_csv['Class'].map(idict)

front_csv.to_csv(fp, index=False)
rear_csv.to_csv(rp, index=False)
test_csv.to_csv(tp, index=False)

testnp = np.load(tnpy)
newnp = []
for index, carm in testnp:
    newnp.append([index, idict[carm]])
np.save(tnpy, np.array(newnp))


