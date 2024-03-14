import os
import torch
import pandas as pd
import scanpy as sc

from SpatialGlue import Train_SpatialGlue

# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'

# fix random seed
#from preprocess import fix_seed
#random_seed=2022
#fix_seed(random_seed)

# the number of clusters
n_clusters = 5 #18

# read data
file_fold = '/home/yahui/anaconda3/work/SpatialGlue_3M/data/Sim_tri/' # please replace 'file_fold' with the download path

adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
adata_omics3 = sc.read_h5ad(file_fold + 'adata_ATAC.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
adata_omics3.var_names_make_unique()

from preprocess import fix_seed, pca, clr_normalize_each_cell, lsi, construct_neighbor_graph
# configure random seed
random_seed=2022
fix_seed(random_seed)

 
# RNA
#sc.pp.filter_genes(adata_omics1, min_cells=10)
#sc.pp.filter_cells(adata_omics1, min_genes=200)
n_protein = adata_omics2.n_vars
  
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
  
adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_protein)
  
# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_protein)
  
# ATAC
sc.pp.highly_variable_genes(adata_omics3, flavor="seurat_v3", n_top_genes=3000)
lsi(adata_omics3, use_highly_variable=False, n_components=n_protein + 1)
adata_omics3.obsm['feat'] = adata_omics3.obsm['X_lsi'].copy()
  
data = construct_neighbor_graph(adata_omics1, adata_omics2, adata_omics3)

# clustering
#clustering(adata_omics1, key='feat', add_key='RNA', n_clusters=5)
#clustering(adata_omics2, key='feat', add_key='protein', n_clusters=5)

#import numpy as np
# flip tissue image
#adata_omics1.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_omics1.obsm['spatial'])).T).T).T
#adata_omics1.obsm['spatial'][:,1] = -1*adata_omics1.obsm['spatial'][:,1]

#adata_omics2.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata_omics2.obsm['spatial'])).T).T).T
#adata_omics2.obsm['spatial'][:,1] = -1*adata_omics2.obsm['spatial'][:,1]

# visualization
import matplotlib.pyplot as plt

# define and train model
model = Train_SpatialGlue(data, device=device, epochs=200)  # spleen [1,5,1,1] thymus [1,10,1,10] lymph node [1,5,1,10]
output = model.train()

# we set 'mclust' as clustering tool by default. Users can also select Leiden and louvain 
tool = 'mclust' # mclust, leiden, and louvain

adata_combined = adata_omics1.copy()
adata_combined.obsm['SpatialGlue'] = output['SpatialGlue']
#print('SpatialGlue:', output['SpatialGlue'])
adata_combined.obsm['feat_pro'] = adata_omics2.obsm['feat'].copy() 
adata_combined.obsm['feat_lsi'] = adata_omics3.obsm['feat'].copy()

# performing PCA
#from preprocess import pca
#adata_combined.obsm['SpatialGlue'] = pca(adata_combined, use_reps='SpatialGlue', n_comps=20)
#print('pca_result:', adata_combined.obsm['SpatialGlue'])

from utils import clustering

# clustering
if tool == 'mclust':
   clustering(adata_combined, key='SpatialGlue', add_key='SpatialGlue', n_clusters=n_clusters, method=tool, use_pca=True)
elif tool in ['leiden', 'louvain']:
   clustering(adata_combined, key='SpatialGlue', add_key='SpatialGlue', n_clusters=n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)
   
# visualization
fig, ax_list = plt.subplots(1, 2, figsize=(12, 5))
sc.pp.neighbors(adata_combined, use_rep='SpatialGlue', n_neighbors=10)
sc.tl.umap(adata_combined)

sc.pl.umap(adata_combined, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=60, show=False)
sc.pl.embedding(adata_combined, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=90, show=False)

#adata_combined.write_h5ad('/home/yahui/anaconda3/work/SpatialGlue_revision/output/Dataset9_Mouse_Brain3/adata_output_H3K27ac.h5ad')

plt.tight_layout(w_pad=0.3)
plt.show()  

