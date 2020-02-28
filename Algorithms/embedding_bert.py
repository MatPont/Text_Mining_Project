# Author: Mathieu Pont

import sys
import pandas as pd
import numpy as np
from bert_serving.client import BertClient
from scipy import sparse, io
from nltk.tokenize import word_tokenize
import re

from config import dataset_path



# Need bert-as-service:
# https://github.com/hanxiao/bert-as-service


def embed_words(vocabulary_file, out_file):
    df = pd.read_csv(vocabulary_file, index_col = 0, keep_default_na = False)
    print(df.shape)
    nrow = df.shape[0]

    bc = BertClient()

    vectors = []

    for word, i in zip(df['0'], range(len(df['0']))):
        if i % 100 == 0:
            print("{} / {} --- {}".format(i + 1, nrow, word))
        tocode = [str(word)]
        code = bc.encode(tocode)
        vectors.append(code[0])
    
    out_df = pd.DataFrame(vectors)
    #out_df.to_csv(out_file)
    csr = sparse.csr_matrix(out_df)
    io.savemat(out_file, {'X': csr})


def embed_docs(documents_file, out_file):
    df = pd.read_csv(documents_file, index_col = 0, keep_default_na = False)
    print(df.shape)
    nrow = df.shape[0]

    bc = BertClient(check_length=False)

    vectors = []
    begin = 0
    try:
        out_df = pd.DataFrame(io.loadmat(out_file)['X'].toarray())
        vectors.append(out_df)
        begin = len(out_df)
        print(begin)
    except:
        pass

    for doc, i in zip(df['Text'], range(len(df['Text']))):            
        if i % 100 == 0:
            print("{} / {}".format(i, nrow - 1))    
        if i < begin:
            continue
        
        doc = str(re.sub("_","",doc))
        tocode = [doc] ; is_tokenized = False
        #tocode = [word_tokenize(doc)] ; is_tokenized = True
        
        code = bc.encode(tocode, is_tokenized = is_tokenized)
        vectors.append(pd.DataFrame([code[0]]))
        
        if i % 500 == 0:
            out_df = pd.concat(vectors)
            out_df.index = np.arange(len(out_df))
            csr = sparse.csr_matrix(out_df)
            io.savemat(out_file, {'X': csr})
    
    out_df = pd.concat(vectors)
    out_df.index = np.arange(len(out_df))
    csr = sparse.csr_matrix(out_df)
    io.savemat(out_file, {'X': csr})



if __name__ == "__main__":    
    if sys.argv[1] == "help":
        print("- embed_words <vocabulary_file> <out_file>")
        print("- embed_words2 <dataset_folder> <dataset>")
        print("\n- embed_docs <documents_file> <out_file>")
        
    if sys.argv[1] == "embed_words":
        embed_words(sys.argv[2], sys.argv[3])
    
    if sys.argv[1] == "embed_words2":
        base_file = dataset_path+"/"+str(sys.argv[2])+"/"+sys.argv[3]+"_preprocessed_"
        vocabulary_file = base_file+"vocabulary.csv"
        out_file = base_file+"embedding.mat"
        embed_words(vocabulary_file, out_file)
    
    if sys.argv[1] == "embed_docs":    
        embed_docs(sys.argv[2], sys.argv[3])
