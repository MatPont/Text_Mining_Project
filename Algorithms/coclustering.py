from coclust.coclustering import CoclustInfo, CoclustMod
from coclust.evaluation.external import accuracy
import sys
import pandas as pd
import numpy as np
from scipy import io
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari

label_file = sys.argv[1]+"_preprocessed.csv"
mat_file = sys.argv[1]+"_preprocessed_"+sys.argv[2]+".mat"

df = pd.read_csv(label_file)
y = np.unique(df['Label'], return_inverse=True)[1]

mat = io.loadmat(mat_file)['X']
print(mat.shape)

no_cluster = len(np.unique(y))
print(no_cluster)



#####################
# CoclustInfo
#####################

model = CoclustInfo(n_row_clusters=no_cluster, n_col_clusters=no_cluster, n_init=20)
model.fit(mat)

print("##############\n# CoclustInfo\n##############")
print(nmi(model.row_labels_, y))
print(ari(model.row_labels_, y))
print(accuracy(model.row_labels_, y))



#####################
# CoclustMod
#####################
model = CoclustMod(n_clusters=no_cluster, n_init=20)
model.fit(mat)

print("##############\n# CoclustMod\n##############")
print(nmi(model.row_labels_, y))
print(ari(model.row_labels_, y))
print(accuracy(model.row_labels_, y))
