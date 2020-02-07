# Author: Mathieu Pont

import sys
import pandas as pd
import json

def separe_text_label(file_name, out_prefix_file_name):
    df = pd.read_csv(file_name, sep="\t")
    df.columns = ['Label', 'Text']
    df.to_csv(out_prefix_file_name+".txt", index=False)
    
def separe_text_label_json(file_name, out_prefix_file_name):
    df_texts = pd.DataFrame()
    df_labels = pd.DataFrame()    
    for line in open(file_name):
        myjson = json.loads(line)
        df_texts = df_texts.append(pd.DataFrame([myjson['raw']]))
        df_labels = df_labels.append(pd.DataFrame([myjson['label']]))
    df = pd.concat([df_labels, df_texts], axis=1)
    df.columns = ['Label', 'Text']
    df.to_csv(out_prefix_file_name+".txt", index=False)
        

if __name__ == "__main__":
    if sys.argv[1] == "separe":
        separe_text_label(sys.argv[2], sys.argv[3])
        
    if sys.argv[1] == "separe_json":
        separe_text_label_json(sys.argv[2], sys.argv[3])

