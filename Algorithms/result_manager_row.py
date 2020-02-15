# Author: Mathieu Pont

from config import result_path, dataset_path
import pandas as pd
from statistics import mean, stdev
import itertools
from shutil import copyfile

data_versions = ["2000", "5000", "5000_AllTag", "5000_50", "5000_50_AllTag"]
mat_versions = ["bow", "tf-idf-l2"]
datasets = ["classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb"]
#algos = ["CoclustInfo", "CoclustMod"]
algos = ["CoclustInfo", "CoclustMod", "CoclustSpecMod"]

#chosen = "bow 5000_50"

def create_string(array):
    array_mean = round(mean(array), 2)
    array_stdev = round(stdev(array), 2)
    return str(array_mean)+" $\pm$ "+str(array_stdev)
    
def print_result(data_version, dataset, mat_version, algo):
    res_file = result_path+"/"+data_version+"/"+dataset+"_"+mat_version+"_"+algo+".txt"
    df = pd.read_csv(res_file, header=None)
    print("#######\n# {} {}\n#######".format(mat_version, data_version))
    print("NMI =", create_string(df.iloc[:,0]))
    print("ARI =", create_string(df.iloc[:,1]))
    print("ACC =", create_string(df.iloc[:,2]))
    return df

all_version = list(map(lambda x: ' '.join(x), itertools.product(mat_versions, data_versions)))
all_version_mean = {newlist: [] for newlist in all_version}

all_best_string = []
all_best_mean = []

# Iterate through each dataset, algorithm, mat_version, data_version to access the results of it
for dataset in datasets:
    print("#####################\n# {}\n#####################".format(dataset))
    for algo in algos:
        best_mean = 0
        best_string = ""
        print("##############\n# {}\n##############".format(algo))
        for mat_version in mat_versions:
            for data_version in data_versions:                
                try:
                    df = print_result(data_version, dataset, mat_version, algo)
                except:
                    continue
                
                mean_algo = mean(pd.concat([df.iloc[:,0], df.iloc[:,1], df.iloc[:,2]]))
                print("MEAN=", round(mean_algo, 2))
                version = mat_version + " " + data_version
                all_version_mean[version].append(mean_algo)
                if mean_algo > best_mean:
                    best_mean = mean_algo
                    best_string = "%8s %11s" % (dataset, algo) + " " + version

        all_best_string.append(best_string)
        all_best_mean.append(best_mean)


# Print best metrics for each dataset and algorithm
print("\n\n\n")
errors = {newlist: 0 for newlist in all_version}
compare = []
for i in range(len(all_best_string)):
    if i % len(algos) == 0:
        print("#####################")
    print(all_best_string[i])
    print(round(all_best_mean[i], 2))
    #print("chosen =", round(all_version_mean[chosen][i], 2))
    for version in all_version_mean:
        errors[version] += (all_best_mean[i] - all_version_mean[version][i]) ** 2



print("\n\n\n")
for version in errors:
    print(version)
    print(round(errors[version], 4))



##########################################
# Get best version for each dataset
##########################################
def save_best_result(best_version, this_dataset):
    # Copy files in the "best" folder
    best_mat_version, best_data_version = best_version.split(" ")
    path = dataset_path+"/"+best_data_version+"/"+this_dataset
    new_path = dataset_path+"/best/"+this_dataset
    files = ["_preprocessed.csv", "_preprocessed_"+best_mat_version+".mat", "_preprocessed_vocabulary.csv"]
    out_files = ["_preprocessed.csv", "_preprocessed.mat", "_preprocessed_vocabulary.csv"]    
    for i in range(len(files)):
        file_name = files[i]
        print(path+file_name)
        copyfile(path+file_name, new_path+out_files[i])


print("\n\n\n")
compare = []
all_dataset_dict = {}
best_version = {}
no_algo = len(algos)
for d in range(len(datasets)):
    dataset_dict = {}
    best_mean = 0
    for version in all_version_mean:
        temp_mean = mean(all_version_mean[version][d*no_algo:d*no_algo+no_algo])
        dataset_dict[version] = temp_mean
        if temp_mean > best_mean:
            best_mean = temp_mean
            best_version[datasets[d]] = version
    all_dataset_dict[datasets[d]] = dataset_dict
    
for d_name in sorted(best_version):
    print(d_name, best_version[d_name], all_dataset_dict[d_name][best_version[d_name]])
    for algo in algos:
        print("##############\n# {}\n##############".format(algo))    
        data_version = best_version[d_name].split(" ")[1]
        mat_version = best_version[d_name].split(" ")[0]
        print_result(data_version, d_name, mat_version, algo)
    save_best_result(best_version[d_name], d_name)






"""print("\n\n\n")
compare = []
for i in range(len(all_best_string)):
    # get value for this version for the other algo
    new_ind = i + (1 if i % 2 == 0 else -1)
    new_version = ' '.join(all_best_string[i].split(" ")[-2:])
    diff = mean([all_best_mean[i], all_version_mean[new_version][new_ind]])
    compare.append((new_version, diff))
for v in compare:
    print(v) 
print()

for i in range(0, len(compare), 2):
    best_version = compare[i][0] if compare[i][1] > compare[i+1][1] else compare[i+1][0]
    best_version_mean = max(compare[i][1], compare[i+1][1])
    this_dataset = datasets[i//2]
    print(this_dataset, best_version, best_version_mean)
    
    save_best_result(best_version, this_dataset)"""
    
        

