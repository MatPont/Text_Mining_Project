# Text_Mining_Project

To reproduce the results of the paper you can do the following steps:

```cd Algorithms/```

```./pipeline_preprocessing.sh```

it will call the "pipeline" execution of ```preprocessing_tools.py``` to do the pre-processing of the raw texts and the "base" execution of ```matrix_tools.py``` to make the bow and tf-idf-l2 matrices.

```./pipeline_coclustering.sh``` 

it will run the co-clustering algorithms (CoclustInfo, CoclustMod and CoclustSpecMod) on all the versions of each dataset with ```coclustering.py``` then keep the best version of each dataset with ```result_manager_row.py``` and finally get the results of the columns clustering with ```coclustering_col.py```.
