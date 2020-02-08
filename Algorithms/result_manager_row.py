# Author: Mathieu Pont

from config import result_path, dataset_path
import pandas as pd
from statistics import mean, stdev

data_versions = ["2000", "5000", "5000_50", "5000_50_AllTag"]
mat_versions = ["bow", "tf-idf-l2"]
datasets = ["classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb"]
algos = ["CoclustInfo", "CoclustMod"]

def create_string(array):
    array_mean = round(mean(array), 2)
    array_stdev = round(stdev(array), 2)
    return str(array_mean)+" $\pm$ "+str(array_stdev)

for dataset in datasets:
    print("#####################\n# {}\n#####################".format(dataset))
    for algo in algos:
        print("##############\n# {}\n##############".format(algo))
        for data_version in data_versions:
            for mat_version in mat_versions:
                res_file = result_path+"/"+data_version+"/"+dataset+"_"+mat_version+"_"+algo+".txt"
                try:
                    df = pd.read_csv(res_file, header=None)                
                except:
                    continue
                print("#######\n# {} {}\n#######".format(data_version, mat_version))
                print("NMI =", create_string(df.iloc[:,0]))
                print("ARI =", create_string(df.iloc[:,1]))
                print("ACC =", create_string(df.iloc[:,2]))
