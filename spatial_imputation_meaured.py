import anndata as ad
from src.CellPLM.utils import set_seed
from src.CellPLM.utils.data import stratified_sample_genes_by_sparsity
from src.CellPLM.pipeline.imputation import ImputationPipeline, ImputationDefaultPipelineConfig, ImputationDefaultModelConfig
import os
import torch
import os
import pandas as pd
import hdf5plugin
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cellplm(PRETRAIN_VERSION, train_data, batch_gene_list, query_data, target_genes):
    
    # Overwrite parts of the default config
    pipeline_config = ImputationDefaultPipelineConfig.copy()
    model_config = ImputationDefaultModelConfig.copy()
    
    pipeline = ImputationPipeline(pretrain_prefix=PRETRAIN_VERSION, 
                                      overwrite_config=model_config,  
                                      pretrain_directory='../ckpt')
    
    pipeline.fit(train_data,
            pipeline_config, 
            split_field = 'split', 
            train_split = 'train',
            valid_split = 'valid',
            batch_gene_list = batch_gene_list, 
            device = device,
            ) 
    
    pred = pipeline.predict(
        query_data,
        pipeline_config, 
        device = device,
    )
    
    results = pipeline.score(
                query_data, 
                evaluation_config = {'target_genes': target_genes},
                label_fields = ['truth'], 
                device = device,
    )
    
    Imp_Genes = pd.DataFrame(pred.cpu(), columns=query_data.var_names)
    
    return Imp_Genes, results



if __name__ == '__main__':
    # DATASET = 'Liver' 
    DATASET = 'Lung'
    
    # load dataset
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

    # sample 25 genes of each group in sparcity boundaries = [0, 0.75, 0.9, 0.95, 1]
    target_genes = stratified_sample_genes_by_sparsity(query_data, seed=11) 
    query_data.obsm['truth'] = query_data[:, target_genes].X.toarray()
    query_data[:, target_genes].X = 0
    train_data = query_data.concatenate(ref_data, join='outer', batch_key=None, index_unique=None)

    train_data.obs['split'] = 'train' # cell/voxel genes
    train_data.obs['split'][train_data.obs['batch']==query_data.obs['batch'][-1]] = 'valid' # - target voxel genes
    train_data.obs['split'][train_data.obs['batch']==ref_data.obs['batch'][-1]] = 'valid' # - reference cell genes
    
    # Specify gene to impute
    query_genes = [g for g in query_data.var.index if g not in target_genes]
    query_batches = list(query_data.obs['batch'].unique())
    ref_batches = list(ref_data.obs['batch'].unique())
    batch_gene_list = dict(zip(list(query_batches) + list(ref_batches),
        [query_genes]*len(query_batches) + [ref_data.var.index.tolist()]*len(ref_batches)))
    
    # training process
    Imp_Genes, results = cellplm(PRETRAIN_VERSION='20231027_85M', train_data=train_data, batch_gene_list=batch_gene_list, query_data=query_data, target_genes=target_genes)
    print(DATASET, " results:", results)
    

# conda activate cellplm     
# CUDA_VISIBLE_DEVICES='0' python spatial_imputation_meaured.py
