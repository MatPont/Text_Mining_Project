data_versions="2000 5000 5000_AllTag 5000_50 5000_50_AllTag"
mat_versions="bow tf-idf-l2"

n_iter=10

for data_version in $data_versions;
do
  for mat_version in $mat_versions;
  do
    echo $data_version $mat_version
    for i in `seq 1 $n_iter`;
    do
      echo $i
      python3 coclustering.py $data_version $mat_version > /dev/null 2>&1
    done
  done
done

python3 result_manager_row.py

python3 coclustering_col.py best
