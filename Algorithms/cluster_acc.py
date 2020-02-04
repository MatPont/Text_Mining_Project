# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 14:31:40 2016

@author: bo
"""

from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
import pandas as pd
import sys
from config import dataset_path, result_path


def acc(ypred, y):
	"""
	Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
	
	ypred: 1-D numpy vector, predicted labels
	y: 1-D numpy vector, ground truth

	The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
	This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
	
	"""
	assert len(y) > 0
	assert len(np.unique(ypred)) == len(np.unique(y))
	
	s = np.unique(ypred)
	t = np.unique(y)
	
	N = len(np.unique(ypred))
	C = np.zeros((N, N), dtype = np.int32)
	for i in range(N):
		for j in range(N):
			idx = np.logical_and(ypred == s[i], y == t[j])
			C[i][j] = np.count_nonzero(idx)
	
	# convert the C matrix to the 'true' cost
	Cmax = np.amax(C)
	C = Cmax - C
	# 
	indices = linear_assignment(C)
	row = indices[:][:, 0]
	col = indices[:][:, 1]
	# calculating the accuracy according to the optimal assignment
	count = 0
	for i in range(N):
		idx = np.logical_and(ypred == s[row[i]], y == t[col[i]] )
		count += np.count_nonzero(idx)
	
	return 1.0*count/len(y)

if __name__ == '__main__':
	"""
	Using accuracy to evaluate clustering is usually not a good idea, the following example shows that 
	even a completely wrong assignment yield accuracy of 0.5.
	
	Consider use more standard metrics, such as NMI or ARI.
	
	"""			  
	#s = np.array([1, 2, 2 ,3, 1, 3])
	#t = np.array([1, 1, 2,2, 3, 3])	
	
	if sys.argv[1] == "help":
		print("- acc <label_file> <cluster_file>")
		print("- acc2 <dataset_name>")
	
	
	if sys.argv[1] == "acc":
		label_file_name = sys.argv[2]
		cluster_file_name = sys.argv[3]
	
	if sys.argv[1] == "acc2":
		dataset_name = sys.argv[2]
		label_file_name = dataset_path+"label/"+dataset_name+"_label.csv"
		cluster_file_name = result_path+dataset_name+"/coclustering/coclustering_labels_row.csv"
	
	if sys.argv[1] == "acc" or sys.argv[1] == "acc2":
		s = np.array(pd.read_csv(label_file_name, index_col=0)).ravel()
		t = np.array(pd.read_csv(cluster_file_name, index_col=0)).ravel()
		
		s -= min(s)
		t -= min(t)
		
		ac = acc(s, t)
		print(ac)
