# Author: Mathieu Pont

import sys
import scipy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from math import log
from scipy import io, sparse
from glove import Corpus
from nltk.tokenize import word_tokenize
from functools import partial

from config import dataset_path
from util import loadmat



# Process text file and return docs and meta_data
def file_to_data(file_name):
	df = pd.read_csv(file_name, header=0, index_col=0)	
	docs = df['Text']
	
	# Manage meta-data (if any)
	meta_data = df.loc[:, df.columns != 'Text']
	
	return docs, meta_data

# Returns dataFrame from docs and meta_data
def data_to_dataFrame(docs, meta_data, vectorizer):
	X = vectorizer.fit_transform(docs) # sparse data
	try:
		df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
	except: # Sometimes MemoryError is raised
		df = pd.DataFrame(None, columns=vectorizer.get_feature_names())
	
	# Manage meta-data (if any)
	"""if meta_data is not None:
		df_meta = meta_data
		df = pd.concat([df, df_meta], axis=1)"""
		
	return df, X

# Whole pipeline text to dataFrame
def file_to_dataFrame(file_name, vectorizer):
	docs, meta_data = file_to_data(file_name)
	df, X = data_to_dataFrame(docs, meta_data, vectorizer)
	return df, X
	
# Return basic bag of words
def file_to_bow(file_name):
	vectorizer = CountVectorizer()
	df, X = file_to_dataFrame(file_name, vectorizer)
	return df, X
	
# Return tf-idf bag of words
def file_to_tfidf(file_name):
	vectorizer = TfidfVectorizer(norm=None, sublinear_tf=True)
	df, X = file_to_dataFrame(file_name, vectorizer)
	return df, X	
	
# Return tf-idf with l2 norm bag of words	
def file_to_tfidf_l2(file_name, norm='l2', sublinear_tf=True, smooth_idf=True):
	print("- norm =",norm)
	if sublinear_tf: print("- sublinear_tf")
	if smooth_idf: print("- smooth_idf")
	vectorizer = TfidfVectorizer(norm=norm, sublinear_tf=sublinear_tf, smooth_idf=smooth_idf)
	df, X = file_to_dataFrame(file_name, vectorizer)
	return df, X	

def file_to_vocabulary(file_name):
	df, X = file_to_bow(file_name)
	vocab = df.columns.values
	return vocab
	
def vocabulary_to_dict(vocabulary_file):
	df = pd.read_csv(vocabulary_file, index_col = 0)
	vocab_dict = {}
	for i, word in zip(range(len(df['0'])), df['0']):
		vocab_dict[word] = i
	return vocab_dict
	



"""=================================
======= CO-OCCURRENCE MATRIX =======
================================="""
def lines_from_text(text_file):
	texts = pd.read_csv(text_file, index_col=0)
	lines = []
	for i, row in texts.iterrows():
		splitted_line = row['Text'].split(".")
		for l in splitted_line:
			token = word_tokenize(l)
			if token != []:
				lines.append(token)
	
	return lines

def co_occ_matrix(lines, window=10):	
	corpus = Corpus()
	corpus.fit(lines, window=window)	
	return corpus.matrix
	
def text_to_co_occ_matrix(text_file, window=10):
	return co_occ_matrix(lines_from_text(text_file), window=window)


def co_occ_matrix_filter(lines, vocabulary_file, window=10):
	# Load vocab file
	df = pd.read_csv(vocabulary_file, index_col = 0)
	
	# Make co-occurrence matrix
	corpus = Corpus()
	corpus.fit(lines, window=window)
	
	mat = corpus.matrix.todense()
	
	# Keep words of vocab
	index = []
	for word in df['0']:
		index.append(corpus.dictionary[word])
	
	mat = mat[index,:]
	mat = mat[:,index]	
	mat = sparse.csr_matrix(mat)
	
	return mat
	
def text_to_co_occ_matrix_filter(text_file, vocabulary_file, window=10):
	return co_occ_matrix_filter(lines_from_text(text_file), vocabulary_file, window=window)
	

# Window-based co-occurrence matrix with centered window (unlike Corpus)
def co_occ_matrix2(lines, vocabulary_file, window=5):
	vocab_dict = vocabulary_to_dict(vocabulary_file)
	vocab_len = len(vocab_dict)
	co_occ_mat = np.zeros((vocab_len,vocab_len))
	
	lines_len = len(lines)	
	for n, line in zip(range(lines_len), lines):
		if n % 1000 == 0:
			print(n," / ",lines_len)

		line_len = len(line)
		for i, word in zip(range(line_len), line):
			window_start = int(max(0, i - window))
			window_stop = int(min(line_len, i + window + 1))

			for j in range(window_start, window_stop):
				if i == j:
					continue					
				if not(str(word) in vocab_dict and str(line[j]) in vocab_dict):
					continue

				ind_i = vocab_dict[word]
				ind_j = vocab_dict[line[j]]
				
				coeff = 1 / abs(j-i)				
				co_occ_mat[ind_i][ind_j] += coeff
				co_occ_mat[ind_j][ind_i] += coeff				

	co_occ_mat = sparse.csr_matrix(co_occ_mat)
	return co_occ_mat

def text_to_co_occ_matrix2(text_file, vocabulary_file, window=10):
	return co_occ_matrix2(lines_from_text(text_file), vocabulary_file, window=window)


def doc_term_to_co_occ(mat_file):
	mat = loadmat(mat_file)
	texts_df = scipy.sparse.csr_matrix(mat)
	
	texts_df[texts_df != 0] = 1
	
	co_occurence_matrix = np.dot(texts_df.T, texts_df)
	
	print(co_occurence_matrix)
	print(co_occurence_matrix.shape)
	
	return co_occurence_matrix	



"""=======================
======= PMI MATRIX =======
======================="""
def compute_pmi(cij, cid, cdj, cdd, default = 0):		
	try:
		Pw = cid #cij / cid
		Pc = cdj #cij / cdj
		Pwc = cij * cdd	#cij / cdd
		ret = log((Pwc) / (Pw * Pc))
	except:
		ret = default
	return ret

def compute_ppmi(pmi):
	return max(pmi, 0)
	
def compute_sppmi(pmi, N = 2):
	return max(pmi - log(N), 0)

# ===========================
# ===== PMI from matrix =====
# ===========================
def xpmi_matrix_mat(matrix, compute_fun=lambda x:x):
	xpmi_matrix = np.zeros(matrix.shape)
	shape_row = matrix.shape[0]
	shape_col = matrix.shape[1]	
	#matrix = normalize(matrix)

	row_mat = matrix.sum(axis = 1)
	row_mat = np.reshape(row_mat, [shape_row, 1])
	col_mat = matrix.sum(axis = 0)
	col_mat = np.reshape(col_mat, [shape_col, 1])
	mat_sum = matrix.sum()

	cx = scipy.sparse.coo_matrix(matrix)
	old_ind = -1
	for i,j,v in zip(cx.row, cx.col, cx.data):
		if old_ind != i:
			if i % 500 == 0:
				print(i)
			old_ind = i
		pmi = compute_pmi(matrix[i,j], row_mat[i], col_mat[j], mat_sum)
		#print(pmi)
		xpmi_matrix[i][j] = compute_fun(pmi)
			
	sparse_xpmi_matrix = scipy.sparse.csr_matrix(xpmi_matrix)
			
	return sparse_xpmi_matrix	

def pmi_matrix_mat(mat):
	return xpmi_matrix_mat(mat)

def ppmi_matrix_mat(mat):
	return xpmi_matrix_mat(mat, compute_ppmi)
	
def sppmi_matrix_mat(mat, N=2):
	return xpmi_matrix_mat(mat, partial(compute_sppmi, N=N))
# ===========================

# =========================
# ===== PMI from file =====
# =========================
def xpmi_matrix(mat_file, compute_fun=lambda x:x):
	matrix = loadmat(mat_file).todense()
	return xpmi_matrix_mat(matrix, compute_fun)

def pmi_matrix(mat_file):
	return xpmi_matrix(mat_file)

def ppmi_matrix(mat_file):
	return xpmi_matrix(mat_file, compute_ppmi)
	
def sppmi_matrix(mat_file, N = 2):
	return xpmi_matrix(mat_file, partial(compute_sppmi, N=N))
# =========================

def count_to_xpmi(mat_file, xpmi="sppmi", N=2):
	co_occ = doc_term_to_co_occ(mat_file)
	fun = globals()[xpmi+"_matrix_mat"]
	if xpmi == "sppmi":
		fun = partial(fun, N=N)
	co_occ.setdiag(0)
	return fun(co_occ.todense())



"""==============================
======= SIMILARITY MATRIX =======
=============================="""
def similarity_matrix_coclustering(embedding_file, word_label_file, vocabulary_file):
	word_vectors = np.matrix(pd.read_csv(embedding_file, index_col=0))
	label = pd.read_csv(word_label_file, index_col=0)
	df_vocab = np.ravel(np.matrix(pd.read_csv(vocabulary_file, index_col = 0)))
	
	label_unique = np.unique(label)
	
	matrices = []
	
	for l in label_unique:
		ind = np.where(label == l)[0]
		words = df_vocab[ind]		
		sim_mat = pd.DataFrame(cosine_similarity(word_vectors[ind]), columns=words, index=words)
		print(sim_mat.shape)
		matrices.append(sim_mat)
	
	return matrices

def similarity_matrix(embedding_file, vocabulary_file):
	#df = pd.read_csv(embedding_file, index_col=0)
	df = loadmat(embedding_file).toarray()
	word_vectors = np.matrix(df)
	df_vocab = np.ravel(np.matrix(pd.read_csv(vocabulary_file, index_col = 0)))
	
	sim_mat = pd.DataFrame(cosine_similarity(word_vectors), columns=df_vocab, index=df_vocab)
	print(sim_mat.shape)
	
	return sim_mat



"""=================
======= MAIN =======
================="""	
if __name__ == '__main__':
	if sys.argv[1] == "help":
		print("- base <csv_file> [<out_bow_file> <out_tf_idf_file> <out_vocab_file>]\n(compute bow, tf-idf-l2 and vocabulary)")
		print("\n- base2 <dataset_name>\n(compute bow, tf-idf-l2 and vocabulary)")
		print("\n- tfidf <text_csv_file> <out_file> [<norm> <sublinear_tf> <smooth_idf>]")
		print("\n- co_occ <text_csv_file> <vocabulary_csv_file> <window_size>\n(compute co-occurrence matrix)")
		print("\n- doc_term_co_occ <doc_term_mat_file> [<out_co_occ_file>]")
		print("\n- count_to_xpmi <mat_file> <pmi/ppmi/sppmi> [<sppmi_N>]")
		print("\n- pmi/ppmi/sppmi <mat_file> [<sppmi_N>]\n(compute pmi/ppmi/sppmi matrix)")
		print("\n- sim_coclustering <embedding_csv_file> <word_label_csv_file> <vocabulary_csv_file> <out_directory>\n(compute similarity matrix for each cluster)")
		print("\n- sim <embedding_csv_file> <vocabulary_csv_file>\n(compute similarity matrix for all word vectors)")



	if sys.argv[1] == "base":
		# 2: raw_csv_file
		in_file = sys.argv[2]
		if len(sys.argv) == 3:
			out_bow_file = in_file[:-4]+"_bow.mat"
			out_tf_idf_file = in_file[:-4]+"_tf-idf-l2.mat"
			out_vocab_file = in_file[:-4]+"_vocabulary.csv"
		else:
			out_bow_file = sys.argv[3]
			out_tf_idf_file = sys.argv[4]
			out_vocab_file = sys.argv[5]
	
	if sys.argv[1] == "base2":	
		# 2: dataset_name
		dataset_name = sys.argv[2]
		in_file = dataset_path+dataset_name+".csv"
		out_bow_file = dataset_path+"mat_files/"+dataset_name+"_bow.mat"
		out_tf_idf_file = dataset_path+"mat_files/"+dataset_name+"_tf-idf-l2.mat"
		out_vocab_file = dataset_path+"vocab/"+dataset_name+"_vocabulary.csv"

	if sys.argv[1] == "base" or sys.argv[1] == "base2":
		# BOW
		print("text to bow...")
		df, X = file_to_bow(in_file)
		print("save mat file...")
		scipy.io.savemat(out_bow_file, {'X' : X})

		# TF-IDF L2
		print("text to tf-idf...")
		df, X = file_to_tfidf_l2(in_file)
		print("save mat file...")
		scipy.io.savemat(out_tf_idf_file, {'X' : X})

		# Vocabulary
		print("text to vocabulary...")
		vocab = file_to_vocabulary(in_file)
		print("save csv file...")
		pd.DataFrame(vocab).to_csv(out_vocab_file)



	if sys.argv[1] == "tfidf":	
		# TF-IDF
		in_file = sys.argv[2]
		out_tf_idf_file = sys.argv[3]
		if len(sys.argv) > 4:
			norm = None if sys.argv[4] == "None" else sys.argv[4]
			sublinear_tf = sys.argv[5] == "1"
			smooth_idf = sys.argv[6] == "1"
			file_to_tfidf_l2 = partial(file_to_tfidf_l2, norm=norm, sublinear_tf=sublinear_tf, smooth_idf=smooth_idf)
		print("text to tf-idf...")
		df, X = file_to_tfidf_l2(in_file)
		print("save mat file...")
		scipy.io.savemat(out_tf_idf_file, {'X' : X})



	if sys.argv[1] == "co_occ":
		# Co-occurrence Matrix
		# 2: text csv file - 3: vocabulary csv file - 4: window size
		print("make lines...")
		lines = lines_from_text(sys.argv[2])
		print("make co-occurrence matrix...")
		#co_occ_mat = co_occ_matrix_filter(lines, sys.argv[3])
		co_occ_mat = co_occ_matrix2(lines, sys.argv[3], window=int(sys.argv[4]))
		print("save co-occurrence matrix...")
		scipy.io.savemat(sys.argv[2][:-4]+"_co-occ_w"+sys.argv[4]+".mat", {'X': co_occ_mat})



	if sys.argv[1] == "doc_term_co_occ":
		# 2: doc_term_mat_file - [3: out_file]
		out_file = sys.argv[2][:-4]+"_co-occ.mat" if len(sys.argv) == 3 else sys.argv[3]
		print("make co-occurrence matrix...")
		co_occ_mat = doc_term_to_co_occ(sys.argv[2])
		print("save co-occurrence matrix...")
		scipy.io.savemat(out_file, {'X': co_occ_mat})



	pmis = ["pmi", "ppmi", "sppmi"]
	if sys.argv[1] in pmis:
		# PMIs Matrix
		# 2: mat file (like co-occurrence matrix)
		fun_name = sys.argv[1]+"_matrix" 
		fun = locals()[fun_name]
		if sys.argv[1] == "sppmi": pmi_matrix = fun(sys.argv[2], N=int(sys.argv[3]))
		else: pmi_matrix = fun(sys.argv[2])
		scipy.io.savemat(sys.argv[2][:-4]+"_"+sys.argv[1]+".mat", {'X': pmi_matrix})
		
		
	
	if sys.argv[1] == "count_to_xpmi":
		N = int(sys.argv[4]) if len(sys.argv) > 4 else None
		pmi_matrix = count_to_xpmi(sys.argv[2], sys.argv[3], N=N)
		scipy.io.savemat(sys.argv[2][:-4]+"_co-occ_"+sys.argv[3]+".mat", {'X': pmi_matrix})		



	if sys.argv[1] == "sim_coclustering":
		# Similarity Matrix for coclustering
		# 2: embedding csv file - 3: word label file - 4: vocabulary file - 5: out directory
		similarity_matrices = similarity_matrix_coclustering(sys.argv[2], sys.argv[3], sys.argv[4])
		for i, m in zip(range(len(similarity_matrices)), similarity_matrices):
			pd.DataFrame(m).to_csv(str(sys.argv[5])+"/coclustering_sim_"+str(i)+".csv")



	if sys.argv[1] == "sim":
		# Similarity Matrix
		# 2: embedding csv file - 3: vocabulary file
		similarity_matrix = similarity_matrix(sys.argv[2], sys.argv[3])
		#pd.DataFrame(similarity_matrix).to_csv(sys.argv[2][:-4]+"_sim-mat.csv")
		csr = sparse.csr_matrix(similarity_matrix)
		io.savemat(sys.argv[2][:-4]+"_sim-mat.mat", {'X': csr})
		
