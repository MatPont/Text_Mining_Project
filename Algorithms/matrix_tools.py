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



"""==============================
======= SIMILARITY MATRIX =======
=============================="""
def similarity_matrix_coclustering_from_file(embedding_file, word_label_file, vocabulary_file):
    #word_vectors = np.matrix(pd.read_csv(embedding_file, index_col=0))
    word_vectors = np.matrix(loadmat(embedding_file).toarray())
    label = pd.read_csv(word_label_file, index_col=0)
    df_vocab = np.ravel(np.matrix(pd.read_csv(vocabulary_file, index_col = 0)))
    
    return similarity_matrix_coclustering(word_vectors, label, df_vocab)

def similarity_matrix_coclustering(word_vectors, label, df_vocab):    
    label_unique = np.unique(label)
    
    matrices = []
    
    for l in label_unique:
        ind = np.where(label == l)[0]
        words = df_vocab[ind]
        sim_mat = pd.DataFrame(cosine_similarity(word_vectors[ind]), columns=words, index=words)
        print(sim_mat.shape)
        matrices.append(sim_mat)
    
    return matrices

def similarity_matrix_from_file(embedding_file, vocabulary_file):
    #df = pd.read_csv(embedding_file, index_col=0)
    df = loadmat(embedding_file).toarray()
    word_vectors = np.matrix(df)
    df_vocab = np.ravel(np.matrix(pd.read_csv(vocabulary_file, index_col = 0)))
    
    return similarity_matrix(word_vectors, df_vocab)

def similarity_matrix(word_vectors, df_vocab):
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
        
