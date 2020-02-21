# Author: Mathieu Pont

import sys
import os
import pandas as pd
import numpy as np
from scipy import io
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from coclust.coclustering import CoclustInfo, CoclustMod, CoclustSpecMod
from coclust.evaluation.external import accuracy
from config import result_path, dataset_path
from util import makedir, loadmat
from matrix_tools import similarity_matrix_coclustering, similarity_matrix


alphas = [0.9, 0.75, 0.5]


#####################
# Functions
#####################
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


def get_pair_words(sim_mat):
    pair_words = sim_mat.where(np.triu(np.ones(sim_mat.shape), 1).astype(np.bool))
    pair_words = pair_words.stack().reset_index()
    pair_words.columns = ['Row','Column','Value']
    pair_words = pair_words.sort_values('Value')
    return pair_words


def compute_column_metrics(column_labels, word_vectors, df_vocab, alphas=alphas):
    accs = []
    all_fp_words = []
    all_fn_words = []
    
    sim_matrix = similarity_matrix(word_vectors, df_vocab)    
    
    labels = np.unique(column_labels)
    for alpha in alphas:
        tp = tn = fp = fn = 0
        for l in labels:
            ind = np.where(column_labels == l)[0]

            # Get the similarity matrix of the cluster
            sim_matrix_label = sim_matrix.iloc[ind,].iloc[:,ind]

            # Get pair words names and score
            fp_pair_words = get_pair_words(sim_matrix_label)
            fp_pair_words = pd.concat([fp_pair_words[:200], fp_pair_words[-200:]])
            all_fp_words.append((alpha, l, fp_pair_words))

            # Compute metrics
            tocompare = np.copy(sim_matrix_label)
            tocompare[np.tril_indices(tocompare.shape[0], 0)] = np.nan            
            tp += (tocompare >= alpha).sum()
            fp += (tocompare < alpha).sum()

        # Filter the sim_matrix by keeping values of words being not in the same cluster
        column_labels = np.array(column_labels)
        matrix_filter = np.matrix(list(map(lambda x: x == column_labels, column_labels)))
        sim_matrix_diff = sim_matrix.copy()
        # It will raise some warnings when we compare with alpha below
        sim_matrix_diff[matrix_filter] = None

        # Compute metrics
        tocompare = np.copy(sim_matrix_diff)
        tocompare[np.tril_indices(tocompare.shape[0], 0)] = np.nan
        tn += (tocompare < alpha).sum()
        fn += (tocompare >= alpha).sum()
        
        # Get pair words names and score
        fn_pair_words = get_pair_words(sim_matrix_diff)
        fn_pair_words = pd.concat([fn_pair_words[:200], fn_pair_words[-200:]])
        all_fn_words.append((alpha, fn_pair_words))

        acc = (tp + tn) / (tp + tn + fp + fn)
        accs.append(acc)

    return accs, all_fp_words, all_fn_words


#####################
# Algo
#####################
if len(sys.argv) > 3 or len(sys.argv) < 2:
    print("Usage: {} data_version [dataset_name]".format(sys.argv[0]))
    exit()

data_version = sys.argv[1]


if len(sys.argv) == 3:
    datasets = [sys.argv[2]]
else:
    datasets = ["classic3", "classic4", "ng5", "ng20", "r8", "r40", "r52", "webkb"]


for dataset in datasets:
    dataset_name = os.path.basename(dataset)
    print("#####################\n# {}\n#####################".format(dataset_name))

    base_file = dataset_path+"/"+data_version+"/"+dataset
    label_file = base_file+"_preprocessed.csv"
    mat_file = base_file+"_preprocessed.mat"
    embedding_file = base_file+"_preprocessed_embedding.mat"
    vocab_file = base_file+"_preprocessed_vocabulary.csv"

    df = pd.read_csv(label_file)
    y = np.unique(df['Label'], return_inverse=True)[1] # as factor

    mat = io.loadmat(mat_file)['X']
    print(mat.shape)

    no_cluster = len(np.unique(y))
    print(no_cluster)    

    word_vectors = np.matrix(loadmat(embedding_file).toarray())
    print(word_vectors.shape)

    df_vocab = np.ravel(np.matrix(pd.read_csv(vocab_file, index_col = 0)))


    algo_pipeline = []
    algo_pipeline.append((CoclustInfo(n_row_clusters=no_cluster, n_col_clusters=no_cluster, n_init=10, max_iter=200), "CoclustInfo"))
    algo_pipeline.append((CoclustMod(n_clusters=no_cluster, n_init=10, max_iter=200), "CoclustMod"))
    algo_pipeline.append((CoclustSpecMod(n_clusters=no_cluster, n_init=10, max_iter=200), "CoclustSpecMod"))

    for model, model_name in algo_pipeline:
        res_nmi, res_ari, res_acc = execute_algo(model, model_name, mat, y)

        res_accs, all_fp_words, all_fn_words = compute_column_metrics(model.column_labels_, word_vectors, df_vocab)

        print("going to save...")
        input()
        # Save results
        out_dir = result_path+"/"+data_version+"/"
        makedir(out_dir)
        base_file = out_dir+dataset+"_"+model_name
        out_file = base_file+"_col.txt"
        content = ', '.join(str(x) for x in res_accs) + "\n"

        myfile = open(out_file, "a")
        myfile.write(content)
        myfile.close()

        df_res = pd.read_csv(out_file, header=None)
        df_res = df_res.max(0)

        out_dir += dataset+"_words/"
        makedir(out_dir)

        for alpha, l, fp_words in all_fp_words:
            # Save results if better than older
            idx = alphas.index(alpha)
            if (res_accs >= df_res)[idx]:
                out_file = out_dir+model_name+"_"+"fp_words_"+str(alpha)+"_"+str(l)+".csv"
                fp_words.to_csv(out_file)

        for alpha, fn_words in all_fn_words:
            # Save results if better than older
            idx = alphas.index(alpha)
            if (res_accs >= df_res)[idx]:
                out_file = out_dir+model_name+"_"+"fn_words_"+str(alpha)+".csv"
                fn_words.to_csv(out_file)
