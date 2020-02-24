dataset_path="../Datasets/"
raw_dataset_path=${dataset_path}"raw_text/"
datasets_name="classic3 classic4 ng5 ng20 r8 r40 r52 webkb"

preprocess_one()
{
    dataset_name=$1
    clean_threshold=$2
    bigram_min_count=$3
    preprocessing_options=$4
    post_out_path=$5
    
    raw_dataset_file=${raw_dataset_path}${dataset_name}.txt
    out_dataset_path=${dataset_path}${clean_threshold}${post_out_path}/
    mkdir $out_dataset_path
    out_dataset_file=${out_dataset_path}${dataset_name}_preprocessed.csv
    
    python3 preprocessing_tools.py pipeline $raw_dataset_file $out_dataset_file $clean_threshold 0 /dev/null $bigram_min_count 10 $preprocessing_options
    python3 matrix_tools.py base $out_dataset_file
}

for dataset_name in datasets_name;
do
    preprocess_one $dataset_name 2000 5 "" ""
    preprocess_one $dataset_name 5000 5 "" ""
    preprocess_one $dataset_name 5000 5 "1 1 1 0 1 1 1 1 1" "_AllTag" # keep all word tags
    preprocess_one $dataset_name 5000 50 "" "_50"
    preprocess_one $dataset_name 5000 50 "1 1 1 0 1 1 1 1 1" "_50_AllTag" # keep all word tags    
done

