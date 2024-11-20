
import os
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

shared_genes = [33, 42, 915, 251, 118, 76, 42, 347, 1000, 154, 141, 118] 


def stratified_sample_genes_by_sparsity(data, boundaries=None, seed=10):
    df = data.to_df()
    zero_rates = 1 - df.astype(bool).sum(axis=0) / df.shape[0]
    if boundaries is None:
        boundaries = [0, 0.75, 0.9, 0.95, 1]
    gene_group = pd.cut(zero_rates, boundaries, labels=False)
    zero_rates = zero_rates.groupby(gene_group, group_keys=False)
    samples = zero_rates.apply(lambda x: x.sample(min(len(x), 25), random_state=seed))
    return list(samples.index)


def datasets_82(dataset_num=2, part=0): 

    sc_ori_8 = sc.read_h5ad("src/mouse_data/dataset{}/dataset{}_seq_{}_8_part_{}.h5ad".format(dataset_num, shared_genes[dataset_num-1],part))
    st_ori_8 = sc.read_h5ad("src/mouse_data/dataset{}/dataset{}_spatial_{}_8_part_{}.h5ad".format(dataset_num, shared_genes[dataset_num-1],part))
    
    sc_ori_2 = sc.read_h5ad("src/mouse_data/dataset{}/dataset{}_seq_{}_2_part_{}.h5ad".format(dataset_num, shared_genes[dataset_num-1],part))
    st_ori_2 = sc.read_h5ad("src/mouse_data/dataset{}/dataset{}_spatial_{}_2_part_{}.h5ad".format(dataset_num, shared_genes[dataset_num-1],part))
    
    return st_ori_8, sc_ori_8, st_ori_2, sc_ori_2


class XDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num = self[list(self.keys())[0]].shape[0]

    def size(self):
        warnings.warn("Deprecated function: Xdict.size().", DeprecationWarning)
        return self._num


def dataset_cellplm(DATASET='Liver'):
    direction='src/human_data/'
    set_seed(11)
    if DATASET == 'Lung':
        query_dataset = os.path.join(direction,'HumanLungCancerPatient2_filtered_ensg.h5ad')
        ref_dataset = os.path.join(direction,'GSE131907_Lung_ensg.h5ad')
        query_data = ad.read_h5ad(query_dataset)
        ref_data = ad.read_h5ad(ref_dataset)

    elif DATASET == 'Liver':
        query_dataset = os.path.join(direction,'HumanLiverCancerPatient2_filtered_ensg.h5ad')
        ref_dataset = os.path.join(direction,'GSE151530_Liver_ensg.h5ad')
        query_data = ad.read_h5ad(query_dataset)
        ref_data = ad.read_h5ad(ref_dataset)
    
    target_genes = stratified_sample_genes_by_sparsity(query_data, seed=11) # This is for reproducing the hold-out gene lists in our paper
    # sample 25 genes of each group in sparcity boundaries = [0, 0.75, 0.9, 0.95, 1]
    query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
    query_data[:, target_genes].X = 0
    train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

    train_data.obs['split'] = 'train' # cell/voxel genes
    train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid' # - target voxel genes
    train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid' # - reference cell genes
    
    dataset = TranscriptomicDataset(train_data, split_field=None, covariate_fields=None, label_fields=None, batch_gene_list=None)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=config['workers'])
    
    return dataloader

        
def draw_data(target_gene, sc_ori, st_ori, Imp_Genes, args):
    
    ## predict/true data correlation
    x = st_ori[:, target_gene].X.toarray().reshape(-1) #### liver
    plt.figure(figsize=(12, 6))
    plt.scatter(x, Imp_Genes[target_gene], color='skyblue', label='Predicted vs True')
    plt.title("Scatter Plot of Predicted vs True Gene Expression {}".format(target_gene))
    plt.xlabel("Ture Gene Expression")
    plt.ylabel("Predict Gene Expression")
    path = os.path.join(args.dir, 'Scatter_data_{}_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)
    
    ## predict/true data
    plt.figure(figsize=(12, 6))
    sns.kdeplot(x, shade=True, color='skyblue')
    plt.title("Gene Expression Density Distribution {}".format(target_gene))
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.xlim(-10, 40)
    path = os.path.join(args.dir, 'data_{}_true_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)

    plt.figure(figsize=(12, 6))
    sns.kdeplot(Imp_Genes[target_gene], shade=True, color='skyblue')
    plt.title("Gene Expression Density Distribution of Lineage {}".format(target_gene))
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.xlim(-10, 40)
    path = os.path.join(args.dir, 'data_{}_predict_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)
    
    ## predict voxel image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_facecolor('black')
    cmap = st_ori[:,target_gene].X.toarray()
    cmap[cmap > np.percentile(cmap,99)] = np.percentile(cmap,99) 
    # ax1.scatter(st_ori.obs['x_coord'], st_ori.obs['y_coord'],s=1,c=cmap) 
    ax1.scatter(st_ori.obs['center_x'], st_ori.obs['center_y'],s=1,c=cmap) 
    ax1.axis('off')
    ax1.set_title('Measured ' + target_gene, fontsize = 16, color='white')
    
    cmap = Imp_Genes[target_gene]
    cmap[cmap > np.percentile(cmap,99)] = np.percentile(cmap,99)
    # ax2.scatter(st_ori.obs['x_coord'], st_ori.obs['y_coord'],s=1,c=cmap)
    ax2.scatter(st_ori.obs['center_x'], st_ori.obs['center_y'],s=1,c=cmap) #### liver
    ax2.axis('off')
    ax2.set_title('Predicted ' + target_gene, fontsize = 16, color='white')
    path = os.path.join(args.dir, 'Voxel_data_{}_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)
    plt.close()
    
    # UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    predict = copy.deepcopy(st_ori)
    predict[:, Imp_Genes.columns].X =  Imp_Genes.values
    predict.obs['label'] = 'st_pre'
    st_ori.obs['label'] = 'st_ori'
    sc_ori.obs['label'] = 'sc_ori'
    color_map = {'st_pre': 'deepskyblue', 'st_ori': 'lightsalmon', 'sc_ori': 'yellowgreen'}
    adata = predict.concatenate(st_ori, sc_ori[:, predict.var_names],join='outer')
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='label',palette=color_map, ax=ax)
    
    plt.title("UMAP of Data {}".format(args.dataset_num), fontsize=18)
    path = os.path.join(args.dir, 'Umap_data_{}.png'.format(args.dataset_num))
    plt.legend(prop={'size': 16}, bbox_to_anchor=(1.0, 0.5), loc='center left', frameon=False)  
    plt.tight_layout()  
    plt.savefig(path)
    plt.close()
    
            
def draw_data_batch(target_gene, sc_ori, st_ori, Imp_Genes, data_name, args):
    ## predict/true data correlation
    x = st_ori[:, target_gene].X.toarray().flatten() 
    plt.figure(figsize=(12, 6))
    plt.scatter(x, Imp_Genes[target_gene], color='skyblue', label='Predicted vs True')
    plt.title("Scatter Plot of Predicted vs True Gene Expression {}".format(target_gene))
    plt.xlabel("Ture Gene Expression")
    plt.ylabel("Predict Gene Expression")
    path = os.path.join(args.dir, 'Scatter_data_{}_gene_{}_batch_{}.png'.format(args.dataset_num, target_gene, args.batch_num))
    plt.savefig(path)
    
    ## predict/true data
    plt.figure(figsize=(12, 6))
    sns.kdeplot(x, shade=True, color='skyblue')
    plt.title("Gene Expression Density Distribution {}_batch_{}".format(target_gene, args.batch_num))
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.xlim(-10, 40)
    path = os.path.join(args.dir, 'data_{}_true_gene_{}_batch_{}.png'.format(args.dataset_num, target_gene, args.batch_num))
    plt.savefig(path)

    plt.figure(figsize=(12, 6))
    sns.kdeplot(Imp_Genes[target_gene], shade=True, color='skyblue')
    plt.title("Gene Expression Density Distribution of Lineage {}_batch_{}".format(target_gene, args.batch_num))
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.xlim(-10, 40)
    path = os.path.join(args.dir, 'data_{}_predict_gene_{}_batch_{}.png'.format(args.dataset_num, target_gene, args.batch_num))
    plt.savefig(path)

    X_spatial_coords = 'x_x' # zhuang
    Y_spatial_coords = 'y_x'

    ## predict voxel image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_facecolor('black')
    cmap = np.ravel(st_ori[:, target_gene].X)
    cmap[cmap > np.percentile(cmap,99)] = np.percentile(cmap,99) 
    ax1.scatter(st_ori.obs[X_spatial_coords], st_ori.obs[Y_spatial_coords],s=1,c=cmap)
    ax1.axis('off')
    ax1.set_title('Measured ' + target_gene, fontsize = 16, color='white')
    
    cmap = Imp_Genes[target_gene]
    cmap[cmap > np.percentile(cmap,99)] = np.percentile(cmap,99)
    ax2.scatter(st_ori.obs[X_spatial_coords], st_ori.obs[Y_spatial_coords],s=1,c=cmap)
    ax2.axis('off')
    ax2.set_title('Predicted ' + target_gene, fontsize = 16, color='white')
    path = os.path.join(args.dir, 'Voxel_data_{}_gene_{}_batch_{}.png'.format(args.dataset_num, target_gene, args.batch_num))
    plt.savefig(path)
    plt.close()
    
    # UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    predict = st_ori.copy()
    predict[:, Imp_Genes.columns].X =  Imp_Genes.values
    predict.obs['label'] = 'st_pre'
    st_ori.obs['label'] = 'st_ori'
    sc_ori.obs['label'] = 'sc_ori'
    color_map = {'st_pre': 'deepskyblue', 'st_ori': 'lightsalmon', 'sc_ori': 'yellowgreen'}
    adata = predict.concatenate(st_ori, sc_ori[:, predict.var_names],join='outer')
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='label',palette=color_map, ax=ax)
    
    plt.title("UMAP of Data {}_batch_{}".format(args.dataset_num, args.batch_num), fontsize=18)
    path = os.path.join(args.dir, 'Umap_data_{}_batch_{}.png'.format(args.dataset_num, args.batch_num))
    plt.legend(prop={'size': 16}, bbox_to_anchor=(1.0, 0.5), loc='center left', frameon=False)  
    plt.tight_layout() 
    plt.savefig(path)
    plt.close()
    
   
