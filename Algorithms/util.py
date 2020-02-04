import os
from scipy import io, sparse
import numpy as np
import random
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from cluster_acc import acc
import pandas as pd



def loadmat(mat_file):
	if mat_file[-3:] == "mat":
		mat = io.loadmat(mat_file)
		name = list(filter(lambda x: sparse.issparse(mat[x]), mat))[0]
		mat = mat[name]
	else:
		df = pd.read_csv(mat_file, index_col=0)
		mat = sparse.csr_matrix(df)
	
	return mat
	
def makedir(folder_out):
	try:
		os.makedirs(folder_out)
	except:
		pass
		
def cartesian_product(*arrays):
	la = len(arrays)
	dtype = np.result_type(*arrays)
	arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
	for i, a in enumerate(np.ix_(*arrays)):
		arr[...,i] = a
	return arr.reshape(-1, la)
	
def shuffle_zip(zip_):
	temp = list(zip_)
	random.shuffle(temp)
	return zip(*temp)
	
def compute_metrics(y, pred, verbose=False):
	nmi = normalized_mutual_info_score(y, pred)
	ari = adjusted_rand_score(y, pred)
	accuracy = acc(y, pred)

	if verbose:
		print("NMI =",nmi)
		print("ARI =",ari)
		print("ACC =",accuracy)

	return {'NMI': nmi, 'ARI': ari, 'ACC': accuracy}
	
def compute_metrics_from_file(y_file, pred_file, verbose=False):
	y = pd.read_csv(y_file, index_col=0).values.ravel()
	pred = pd.read_csv(pred_file, index_col=0).values.ravel()
	return compute_metrics(y, pred, verbose)
	
def write_file(file_name, content, mode="a"):
	my_file = open(file_name, mode)
	my_file.write(content)
	my_file.close()
