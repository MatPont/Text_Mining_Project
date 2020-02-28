# Author: Mathieu Pont

import sys
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tag.perceptron import PerceptronTagger
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from config import dataset_path



lemmatizer = WordNetLemmatizer()
tagger = PerceptronTagger()
pos_tag = lambda x : tagger.tag([x])


preprocessing_steps = ["remove_punctuation", "lowercase", "remove_numbers", "remove_stop_words", 
                        "keep_only_nouns", "lemmatization", "remove_infrequent_words", 
                        "keep_most_frequent_words", "make_bigrams"]


tag_to_keep = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS']
#tag_to_keep = ['NN', 'NNS', 'NNP', 'NNPS']

preprocessed_dict = {} # save the preprocessing of each word to avoid re-computing it

def countWordsOnTexts(df):
    wordCounter = Counter()
    for i, row in df.iterrows():
        text = row["Text"]
        for word in word_tokenize(text):
            wordCounter[word] += 1

    return wordCounter

def cleanseData(df, threshold, out_vocab_file, vocab):
    print('Cleaned Vocabulary size: ' + str(len(vocab)))
    f = open(out_vocab_file, 'w' )
    f.write(repr(vocab))
    f.close()
    frames = []

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i)
        text = row["Text"]
        
        new_text = []
        for word in word_tokenize(text):
            if word in vocab:
                new_text.append(word)
        text = ' '.join(new_text)
        
        if text != text or text == '':
            continue
            
        row['Text'] = text
        frames.append(pd.DataFrame(row).T)

    df = pd.concat(frames)
    df.index = np.arange(len(df))
    print(df.shape)
    
    return df,vocab

def cleanseData_below_threshold(df, threshold, out_vocab_file):
    counter = countWordsOnTexts(df)
    vocab = {x : counter[x] for x in counter if counter[x] >= threshold}
    return cleanseData(df, threshold, out_vocab_file, vocab)
    
def cleanseData_most_common(df, threshold, out_vocab_file):
    counter = countWordsOnTexts(df).most_common(threshold)
    vocab = {word : freq for word, freq in counter}
    return cleanseData(df, threshold, out_vocab_file, vocab)

# clean_fun should be "cleanseData_most_common" or "cleanseData_below_threshold"
def cleanAndSaveData(in_file_name, out_file_name, threshold, out_vocab_file, clean_fun):
    df = pd.read_csv(in_file_name, header=0, index_col=0)
    df,vocab = clean_fun(df, threshold, out_vocab_file)
    df.to_csv(out_file_name)


def preprocessDataset(in_file_name, out_file_name, options):
    df = pd.read_csv(in_file_name)
    df.dropna(inplace=True)

    frames = []

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(i)
        text = row['Text']
        text = preprocess(word_tokenize(text), options)

        if text != text or text == '':
            continue

        row['Text'] = text
        frames.append(pd.DataFrame(row).T)

    df = pd.concat(frames)
    df.index = np.arange(len(df))
    print(df.shape)
    
    counter = countWordsOnTexts(df)
    print('Preprocessed Vocabulary size: ' + str(len(counter)))

    df.to_csv(out_file_name)

# Return correct pos for lemmatization
def get_wordnet_pos(tag):
    tag = tag[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess(words, options):
    new_words = []
    for word in words:
        temp = word
        
        # Remove copyright at the end of the text (specific preprocessing)
        #if temp == "Copyright" or temp == "©" or temp == "Copyright©" or re.match(r"©2.*", temp): 
            #break
        
        if not word in preprocessed_dict:
            # Keep dot for sentence separation
            #if temp != '.':
                preprocessed_dict[word] = ''
                # Remove from array punctuation words
                if options["remove_punctuation"]:            
                    temp = re.sub(r'[^\w\s]', '', temp)

                # To lowercase
                if options["lowercase"]:
                    temp = temp.lower()

                # Remove line breaks
                temp = temp.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('_', ' ')

                # Remove empty words
                if temp == "":
                    continue

                # Remove numbers
                if options["remove_numbers"]:
                    if temp.isdigit():
                        continue

                # Remove stop words
                if options["remove_stop_words"]:            
                    if temp in stopwords.words('english'):
                        continue

                # Get pos_tag of the word
                if options["keep_only_nouns"] or options["lemmatization"]:
                    tag = pos_tag(temp)

                # Remove non-noun words
                if options["keep_only_nouns"]:        
                    if not tag[0][1] in tag_to_keep:
                        continue

                # Lemmatization
                if options["lemmatization"]:
                    temp = lemmatizer.lemmatize(temp, get_wordnet_pos(tag)) # complete lemmatization
                    #temp = lemmatizer.lemmatize(temp) # fast lemmatization but not perfect
                
                """# Stemming
                if options["stemming"]:            
                    stemmer = SnowballStemmer('english')
                    temp = stemmer.stem(temp)
                """
                preprocessed_dict[word] = temp
        else:
            temp = preprocessed_dict[word]

        # Remove empty words
        if temp == '':
            continue

        new_words.append(temp)
        
    # Return a single string with preprocessed text
    return ' '.join(str(x) for x in new_words)
    
    
def make_bigram(text_csv_file, out_file_name, out_vocab_file_name, min_count=50, threshold=10):
    df = pd.read_csv(text_csv_file, index_col=0)
    
    all_tokens = []
    
    for text in df['Text']:
        #all_tokens.append(word_tokenize(text))
        all_tokens.append(word_tokenize(text.replace(".","")))
        
    bigram = Phrases(all_tokens, min_count=min_count, threshold=threshold, delimiter=b'_')
    bigram_phraser = Phraser(bigram)
    
    new_texts = []
    for tokens in all_tokens:
        new_texts.append(' '.join(str(x) for x in bigram_phraser[tokens]))
        
    df_meta = df.loc[:, df.columns != 'Text']
    df = pd.DataFrame(data={'Text': new_texts})
    df = pd.concat([df, df_meta], axis=1)
    df.index = np.arange(len(df))
    print(df.shape)
        
    counter = countWordsOnTexts(df)
    #print(counter)
    print('Bigram Vocabulary size: ' + str(len(counter)))
    
    f = open(out_vocab_file_name, 'w')
    f.write(repr(counter))
    f.close()
        
    df.to_csv(out_file_name)
        
    return df


def make_options_dict(tab=np.ones(len(preprocessing_steps))):
    mydict = {}
    for step, i in zip(preprocessing_steps, range(len(preprocessing_steps))):
        mydict[step] = (int(tab[i]) == 1)
    return mydict

def make_options(sys_argv, n):
    if len(sys_argv) != n:
        options = make_options_dict(sys_argv[n:len(sys_argv)])
    else: 
        options = make_options_dict()
    return options



if __name__ == '__main__':
    if sys.argv[1] == "help":
        print("- preprocess <raw_text_csv_file> <preprocessed_out_csv_file> [<preprocessing_options>]")
        print("\n- preprocessClean <raw_text_csv_file> <preprocessed_out_csv_file> <clean_out_csv_file> <clean_threshold> <vocab_out_file> [<preprocessing_options>]")
        print("\n- clean <preprocessed_csv_file> <clean_out_csv_file> <clean_threshold> <vocab_out_file>")
        print("\n- bigram <text_csv_file> <bigram_min_count> <bigram_score_threshold>")
        print("\n- pipeline <in_text_csv_file> <out_text_csv_file> <clean_infrequent_threshold> <clean_frequent_threshold> <vocab_out_file> <bigram_min_count> <bigram_score_threshold> [<preprocessing_options>]")
        print("\n- pipeline2 <dataset_name> <clean_infrequent_threshold> <clean_frequent_threshold> <bigram_min_count> <bigram_score_threshold> [<preprocessing_options>]")
        
        print("\n=====================")
        print("\n- <preprocessing_options> should be a sequence of 0 or 1, one number for each of the following options:")
        print(' '.join("<"+str(x)+">" for x in preprocessing_steps))



    if sys.argv[1] == "preprocess":
        # 2: raw_text_csv_file - 3: preprocessed_out_csv_file - 4+: preprocessing_options 
        options = make_options(sys.argv, 4)
        preprocessDataset(sys.argv[2], sys.argv[3], options)



    if sys.argv[1] == "preprocessClean":
        # 2: raw_text_csv_file - 3: preprocessed_out_csv_file - 4: clean_out_csv_file
        # 5: clean_threshold - 6: vocab_out_file  - 7+: preprocessing_options
        options = make_options(sys.argv, 7)
        preprocessDataset(sys.argv[2], sys.argv[3], options)
        fun = cleanseData_below_threshold
        cleanAndSaveData(sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], clean_fun=fun)



    if sys.argv[1] == "clean":
        # 2: preprocessed_csv_file - 3: clean_out_csv_file - 4: clean_threshold
        # 5: vocab_out_file
        fun = cleanseData_below_threshold
        cleanAndSaveData(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], clean_fun=fun)



    if sys.argv[1] == "bigram":
        # 2: text_csv_file - 3: bigram_min_count - 4: bigram_score_threshold
        file_name = sys.argv[2][:-4]+"_b"+str(sys.argv[3])+"-"+str(sys.argv[4])
        out_vocab = file_name+"_vocab.json"
        out_file_name = file_name+".csv"
        make_bigram(sys.argv[2], out_file_name, out_vocab, int(sys.argv[3]), int(sys.argv[4]))



    if sys.argv[1] == "pipeline":
        # 2: raw_text_csv_file - 3: preprocessed_out_csv_file - 4: clean_infrequent_threshold
        # 5: clean_frequent_threshold - 6: vocab_out_file - 7: bigram_min_count
        # 8: bigram_score_threshold - 9+: preprocessing_options    
        options = make_options(sys.argv, 9)
        in_file = sys.argv[2]
        out_file = sys.argv[3]
        vocab_file = sys.argv[6]
        clean_infrequent_threshold = int(sys.argv[4])
        clean_frequent_threshold = int(sys.argv[5])
        bigram_min_count = int(sys.argv[7])
        bigram_score_threshold = int(sys.argv[8])

    if sys.argv[1] == "pipeline2":
        # 2: dataset_name - 3: clean_infrequent_threshold - 4: clean_frequent_threshold
        # 5: bigram_min_count - 6: bigram_score_threshold - 7+: preprocessing_options
        options = make_options(sys.argv, 7)
        dataset_name = sys.argv[2]
        in_file = dataset_path+dataset_name+".txt"
        out_file = dataset_path+dataset_name+"_preprocessed.csv"
        vocab_file = dataset_path+"vocab/"+dataset_name+"_preprocessed_vocab.json"
        clean_infrequent_threshold = int(sys.argv[3])
        clean_frequent_threshold = int(sys.argv[4])
        bigram_min_count = int(sys.argv[5])
        bigram_score_threshold = int(sys.argv[6])

    if sys.argv[1] == "pipeline" or sys.argv[1] == "pipeline2":
        print("basic preprocessing...")
        preprocessDataset(in_file, out_file, options)
        # -------
        if options["make_bigrams"]:
            print("make bigrams...")
            make_bigram(out_file, out_file, vocab_file, bigram_min_count, bigram_score_threshold)
        # -------
        if options["remove_infrequent_words"] and clean_infrequent_threshold != 0:
            print("remove infrequent words...")
            fun = cleanseData_below_threshold
            cleanAndSaveData(out_file, out_file, clean_infrequent_threshold, vocab_file, clean_fun=fun)
        # -------
        if options["keep_most_frequent_words"] and clean_frequent_threshold != 0:
            print("keep most frequent words...")
            fun = cleanseData_most_common
            cleanAndSaveData(out_file, out_file, clean_frequent_threshold, vocab_file, clean_fun=fun)



"""
"""










    
"""
df_t = pd.DataFrame(data={'Text': [text]})
df_t_meta = pd.DataFrame(row[row.index != 'Text']).T
df_t_meta.index = [0]
df_t = pd.concat([df_t, df_t_meta], axis=1)
frames.append(df_t)"""        
