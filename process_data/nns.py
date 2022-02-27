from lopq import LOPQModel, LOPQSearcher
import os
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="/mnt/e/data/", type=str,
                        help="Path of the output dir.")
parser.add_argument("--dimension", default=64, type=int)
parser.add_argument("--k", default=20, type=int)
args = parser.parse_args()

modelp = os.path.join(args.save_path)
dbp = os.path.join(args.dpath, "database.npy")
imagep = os.path.join(args.dpath, "tdatabase.npy")
classp = os.path.join(args.dpath, "model_num.json")
resp = os.path.join(args.dpath, "mAP.json")

db = np.load(dbp)
testdb = np.load(imagep)
md = json.load(open(classp, "r"))
testp = os.path.join(args.dpath, "test.npy")
testset = np.load(testp)
testinput = testset[:, 0]
test_class = testset[:, 1]
# Define a model and fit it to data
model = LOPQModel(V=8, M=args.dimension, subquantizer_clusters=256)
model.fit(db)

# Compute the LOPQ codes for a vector
# code = model.predict(x)

# Create a searcher to index data with the model
searcher = LOPQSearcher(model)
searcher.add_data(db)

# Retrieve ranked nearest neighbors
nns = searcher.search(x, quota=args.k)