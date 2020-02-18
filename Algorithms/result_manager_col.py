# Author: Mathieu Pont

from config import result_path, dataset_path
import pandas as pd
import sys
from statistics import mean, stdev

datasets = ["classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb"]
algos = ["CoclustInfo", "CoclustMod", "CoclustSpecMod"]

alphas = [0.9, 0.75, 0.5]

for dataset in datasets:
    print(dataset)
    base_file = result_path+"/best/"+dataset
    res = {i: [] for i in range(3)}
    for algo in algos:
        res_file = base_file+"_"+algo+"_col.txt"
        df = pd.read_csv(res_file, header=None)
        for i in range(3):
            res_i = str(round(mean(df.iloc[:,i]), 2))
            std = round(stdev(df.iloc[:,i]), 2)
            if std != 0.:
                res_i += " $\pm$ " + str(std)
            res[i].append(res_i)
    for name in res:
        post = "& $\\alpha=" + str(alphas[name]) + "$" #& $\alpha=0.75$
        print(post + " & " + ' & '.join(res[name]) + " \\\\")
