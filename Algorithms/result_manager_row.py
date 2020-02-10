# Author: Mathieu Pont

from config import result_path, dataset_path
import pandas as pd
from statistics import mean, stdev
import itertools

data_versions = ["2000", "5000", "5000_AllTag", "5000_50", "5000_50_AllTag"]
mat_versions = ["bow", "tf-idf-l2"]
datasets = ["classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb"]
algos = ["CoclustInfo", "CoclustMod"]

chosen = "bow 5000_50"

def create_string(array):
    array_mean = round(mean(array), 2)
    array_stdev = round(stdev(array), 2)
    return str(array_mean)+" $\pm$ "+str(array_stdev)

all_version = list(map(lambda x: ' '.join(x), itertools.product(mat_versions, data_versions)))
all_version_mean = {newlist: [] for newlist in all_version}

all_string = []
all_mean = []

for dataset in datasets:
    print("#####################\n# {}\n#####################".format(dataset))
    for algo in algos:
        best_mean = 0
        best_string = ""
        print("##############\n# {}\n##############".format(algo))
        for mat_version in mat_versions:
            for data_version in data_versions:
                res_file = result_path+"/"+data_version+"/"+dataset+"_"+mat_version+"_"+algo+".txt"
                try:
                    df = pd.read_csv(res_file, header=None)                
                except:
                    continue
                print("#######\n# {} {}\n#######".format(mat_version, data_version))
                print("NMI =", create_string(df.iloc[:,0]))
                print("ARI =", create_string(df.iloc[:,1]))
                print("ACC =", create_string(df.iloc[:,2]))
                
                mean_algo = mean(pd.concat([df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]]))
                print("MEAN=", round(mean_algo, 2))
                version = mat_version + " " + data_version
                all_version_mean[version].append(mean_algo)
                if mean_algo > best_mean:
                    best_mean = mean_algo
                    best_string = "%8s %11s" % (dataset, algo) + " " + version

        all_string.append(best_string)
        all_mean.append(best_mean)

print("\n\n\n")
errors = {newlist: 0 for newlist in all_version}
for i in range(len(all_string)):
    print(all_string[i])
    print(round(all_mean[i], 2))
    print(round(all_version_mean[chosen][i], 2))
    for version in all_version_mean:
        errors[version] += (all_mean[i] - all_version_mean[version][i]) ** 2

print("\n\n\n")
for version in errors:
    print(version)
    print(round(errors[version], 4))

    
    
