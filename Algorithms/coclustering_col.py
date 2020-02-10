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
from util import makedir, loadmat
from matrix_tools import similarity_matrix_coclustering, similarity_matrix



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


def compute_column_metrics(column_labels, word_vectors, df_vocab, alphas=[0.9, 0.75, 0.5]):
    accs = []
    
    sim_matrix = similarity_matrix(word_vectors, df_vocab)    
    
    labels = np.unique(column_labels)
    for alpha in alphas:
        tp = tn = fp = fn = 0
        for l in labels:
            ind = np.where(column_labels == l)[0]
            # Get the similarity matrix of the cluster
            sim_matrix_label = sim_matrix.iloc[ind,].iloc[:,ind]
            print(sim_matrix_label)
            # Get pair words names and score
            pair_words = sim_matrix_label.where(np.triu(np.ones(sim_matrix_label.shape), 1).astype(np.bool))
            pair_words = pair_words.stack().reset_index()
            pair_words.columns = ['Row','Column','Value']
            pair_words = pair_words.sort_values('Value')

            # Compute metrics
            tp += (np.triu(sim_matrix_label, 1) >= alpha).sum()
            fp += (np.triu(sim_matrix_label, 1) < alpha).sum()
            
        # Filter the sim_matrix by keeping values of words being not in the same cluster
        column_labels = np.array(column_labels)
        matrix_filter = np.matrix(list(map(lambda x: x == column_labels, column_labels)))
        sim_matrix_diff = sim_matrix.copy()
        sim_matrix_diff[matrix_filter] = None
        # Compute metrics
        tn += (np.triu(sim_matrix_diff, 1) < alpha).sum()
        fn += (np.triu(sim_matrix_diff, 1) >= alpha).sum()

        acc = (tp + tn) / (tp + tn + fp + fn)
        accs.append(acc)
    
    return accs



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
    embedding_file = base_file+"_preprocessed_embedding.mat"
    vocab_file = base_file+"_preprocessed_vocabulary.csv"

    df = pd.read_csv(label_file)
    y = np.unique(df['Label'], return_inverse=True)[1] # as factor

    mat = io.loadmat(mat_file)['X']
    print(mat.shape)
    
    word_vectors = np.matrix(loadmat(embedding_file).toarray())
    print(word_vectors.shape)
    
    df_vocab = np.ravel(np.matrix(pd.read_csv(vocab_file, index_col = 0)))

    no_cluster = len(np.unique(y))
    print(no_cluster)
    
    

    algo_pipeline = []
    algo_pipeline.append((CoclustInfo(n_row_clusters=no_cluster, n_col_clusters=no_cluster, n_init=10, max_iter=200), "CoclustInfo"))
    algo_pipeline.append((CoclustMod(n_clusters=no_cluster, n_init=10, max_iter=200), "CoclustMod"))

    for model, model_name in algo_pipeline:
        res_nmi, res_ari, res_acc = execute_algo(model, model_name, mat, y)
        
        res_accs = compute_column_metrics(model.column_labels_, word_vectors, df_vocab)
        
        # Save results
        out_dir = result_path+"/"+data_version+"/"
        makedir(out_dir)
        out_file = out_dir+dataset+"_"+mat_version+"_"+model_name+"_col.txt"
        content = ', '.join(str(x) for x in res_accs) + "\n"
        
        myfile = open(out_file, "a")
        myfile.write(content)
        myfile.close()
