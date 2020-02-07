# Author: Mathieu Pont

import sys
import os
import pandas as pd
import numpy as np
from scipy import io
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from coclust.coclustering import CoclustInfo, CoclustMod
from coclust.evaluation.external import accuracy
from config import result_path, dataset_path



def execute_algo(model, model_name, X, y, verbose=True):
    print("##############\n# {}\n##############".format(model_name))
    model.fit(X)
    res_nmi = nmi(model.row_labels_, y)
    res_ari = ari(model.row_labels_, y)
    res_acc = accuracy(model.row_labels_, y)
    if verbose:
        print("NMI =", res_nmi)
        print("ARI =", res_ari)
        print("ACC =", res_acc)
    return res_nmi, res_ari, res_acc



if len(sys.argv) > 4 or len(sys.argv) < 3:
    print("Usage: {} data_version mat_version [dataset_name]".format(sys.argv[0]))

data_version = sys.argv[1]
mat_version = sys.argv[2]

if len(sys.argv) == 4:
    datasets = [sys.argv[3]]
else:
    datasets = ["classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb"]


for dataset in datasets:
    dataset_name = os.path.basename(dataset)
    print("#####################\n# {}\n#####################".format(dataset_name))

    base_file = dataset_path+"/"+data_version+"/"+dataset
    label_file = base_file+"_preprocessed.csv"
    mat_file = base_file+"_preprocessed_"+mat_version+".mat"

    df = pd.read_csv(label_file)
    y = np.unique(df['Label'], return_inverse=True)[1] # as factor

    mat = io.loadmat(mat_file)['X']
    print(mat.shape)

    no_cluster = len(np.unique(y))
    print(no_cluster)

    algo_pipeline = []
    algo_pipeline.append((CoclustInfo(n_row_clusters=no_cluster, n_col_clusters=no_cluster, n_init=10, max_iter=200), "CoclustInfo"))
    algo_pipeline.append((CoclustMod(n_clusters=no_cluster, n_init=10, max_iter=200), "CoclustMod"))

    for model, model_name in algo_pipeline:
        res_nmi, res_ari, res_acc = execute_algo(model, model_name, mat, y)
        out_file = result_path+"/"+data_version+"/"+dataset+"_"+mat_version+"_"+model_name+".txt"
        content = str(res_nmi)+", "+str(res_ari)+", "+str(res_acc)+"\n"
        myfile = open(out_file, "a")
        myfile.write(content)
        myfile.close()
