import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import entropy
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import scale
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import copy
import torch
import os
from tqdm import tqdm
from model import *
import pdb
import torch.nn.functional as F
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from matplotlib.font_manager import FontProperties
# from tensorboardX import SummaryWriter
import pickle
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')


def GAP(st_ori, st_target, sc_ori, st_embed_uce2, sc_embed_uce2, st_embed_uce, sc_embed_uce, args, genes_to_predict=None):

    # list of gene names missing from the statial data
    if genes_to_predict is GAP.__defaults__[0]:
        print('predict all missing genes\n')
        genes_to_predict = np.setdiff1d(sc_ori.var_names,st_ori.var_names)
    valid_genes_to_predict = [gene for gene in genes_to_predict if gene in sc_ori.var_names]
    Y_train = sc_ori[:, valid_genes_to_predict]

    # unmeasured statial genes expression to be predicted
    Imp_Genes = pd.DataFrame(np.zeros((st_ori.shape[0],len(valid_genes_to_predict))),columns=genes_to_predict)
    Dic = np.zeros((sc_ori.shape[0],st_ori.shape[0]))
    
    # data prepare
    # ori data
    sc_ori_scaled = pd.DataFrame(sc_ori.X, columns=sc_ori.var_names)
    st_ori_scaled = pd.DataFrame(st_ori.X, columns=st_ori.var_names) # index=axis_index,

    # shared ori data
    sc_intsec_scaled = sc_ori_scaled[np.intersect1d(sc_ori_scaled.columns,st_ori_scaled.columns)]
    st_ori_scaled = st_ori_scaled[sc_intsec_scaled.columns]

    scaler = MinMaxScaler(feature_range=(0, 1))

    # GAP
    sc_input = torch.tensor(scaler.fit_transform(sc_intsec_scaled.values)).to(torch.float32).to(device) 
    st_input = torch.tensor(scaler.fit_transform(st_ori_scaled.values)).to(torch.float32).to(device)
    
    model  = DicModel(ori_dim=st_ori_scaled.shape[1], sc_input=sc_input, st_input=st_input, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 

    for epoch in tqdm(range(args.epoch)): 
        # pdb.set_trace()
        x_input, y_input, sc_out, st_out, cell2voxel, y_out, M = model(sc_input, st_input)

        l_align =  F.mse_loss(sc_out.sum(), sc_input.sum()) + F.mse_loss(st_out.sum(), st_input.sum()) \
                - F.cosine_similarity(sc_out, x_input, dim=0).mean() - F.cosine_similarity(st_out, y_input, dim=0).mean() 
                
           
        l_impu = F.l1_loss(torch.matmul(M, sc_input), st_input) 
        # l_M = F.l1_loss(M.sum(dim=1),torch.ones_like(M.sum(dim=1)))
        l_reco = - F.cosine_similarity(y_out, cell2voxel, dim=0).mean() - F.cosine_similarity(y_out, cell2voxel, dim=1).mean() 
        l_proj = l_impu + l_reco #+ l_M
        
        loss =   l_align + args.proj * l_proj
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            D = M.double().cpu().detach().numpy()

    # Imp_Genes[:]= np.dot(D, Y_train.X)
    # pdb.set_trace()
    Imp_Genes[:]= np.dot(D, scaler.fit_transform(Y_train.X))
    Dic = D
    # print(Imp_Genes.values)
    # print(st_ori.X)

    return Imp_Genes, Dic  


# UCE embedding nan for dataset 8
def datasets_UCE_4(dataset_num=2):
    
    '''
    sc data: obs: 'n_genes'(cells); var: 'n_cells'(genes)
    Spatial data: obs: 'x_coord', 'y_coord', 'n_genes'(number); var: 'n_cells'(genes)
    embed data: 33 --> ffeature = 512 scGPT / 1280 UCE
    '''
    if dataset_num==1:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset1_seq_246.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset1_spatial_246.h5ad")
        st_ori.obs['x_coord'] = st_ori.obs['x']
        st_ori.obs['y_coord'] = st_ori.obs['y']
        del st_ori.obs['x']
        del st_ori.obs['y']
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset1_seq_246_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset1_spatial_246_uce_adata.h5ad")
    elif dataset_num==2:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset2_seq_33.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset2_spatial_33.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset2_seq_33_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset2_spatial_33_uce_adata.h5ad")
    elif dataset_num==3:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset3_seq_42.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset3_spatial_42.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset3_seq_42_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset3_spatial_42_uce_adata.h5ad")
    elif dataset_num==4:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset4_seq_1000.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset4_spatial_1000.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset4_seq_1000_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset4_spatial_1000_uce_adata.h5ad")
    elif dataset_num==5:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset5_seq_915.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset5_spatial_915.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset5_seq_915_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset5_spatial_915_uce_adata.h5ad")
    elif dataset_num==6:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset6_seq_251.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset6_spatial_251.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset6_seq_251_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset6_spatial_251_uce_adata.h5ad")
    elif dataset_num==7:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset7_seq_118.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset7_spatial_118.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset7_seq_118_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset7_spatial_118_uce_adata.h5ad")
    elif dataset_num==8:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset8_seq_84.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset8_spatial_84.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset8_seq_84_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset8_spatial_84_uce_adata.h5ad")
    elif dataset_num==9:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset9_seq_76.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset9_spatial_76.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset9_seq_76_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset9_spatial_76_uce_adata.h5ad")
    elif dataset_num==10:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset10_seq_42.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset10_spatial_42.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset10_seq_42_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset10_spatial_42_uce_adata.h5ad")
    elif dataset_num==11:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset11_seq_347.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset11_spatial_347.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset11_seq_347_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset11_spatial_347_uce_adata.h5ad")
    elif dataset_num==12:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset12_seq_1000.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset12_spatial_1000.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset12_seq_1000_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset12_spatial_1000_uce_adata.h5ad")
    elif dataset_num==13:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset13_seq_154.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset13_spatial_154.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset13_seq_154_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset13_spatial_154_uce_adata.h5ad")
    elif dataset_num==14:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset14_seq_981.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset14_spatial_981.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset14_seq_981_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset14_spatial_981_uce_adata.h5ad")
    elif dataset_num==15:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset15_seq_141.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset15_spatial_141.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset15_seq_141_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset15_spatial_141_uce_adata.h5ad")
    elif dataset_num==16:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset16_seq_118.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset16_spatial_118.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset16_seq_118_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-4/dataset16_spatial_118_uce_adata.h5ad")
    else:
        print('Add your datasets')
    
    return sc_ori, st_ori, sc_embed, st_embed 


def datasets_UCE_33(dataset_num=2):
    '''
    sc data: obs: 'n_genes'(cells); var: 'n_cells'(genes)
    Spatial data: obs: 'x_coord', 'y_coord', 'n_genes'(number); var: 'n_cells'(genes)
    embed data: 33 --> ffeature = 512 scGPT / 1280 UCE
    '''
    if dataset_num==1:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset1_seq_246.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset1_spatial_246.h5ad")
        st_ori.obs['x_coord'] = st_ori.obs['x']
        st_ori.obs['y_coord'] = st_ori.obs['y']
        del st_ori.obs['x']
        del st_ori.obs['y']
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset1_seq_246_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset1_spatial_246_uce_adata.h5ad")
    elif dataset_num==2:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset2_seq_33.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset2_spatial_33.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset2_seq_33_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset2_spatial_33_uce_adata.h5ad")
    elif dataset_num==3:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset3_seq_42.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset3_spatial_42.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset3_seq_42_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset3_spatial_42_uce_adata.h5ad")
    elif dataset_num==4:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset4_seq_1000.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset4_spatial_1000.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset4_seq_1000_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset4_spatial_1000_uce_adata.h5ad")
    elif dataset_num==5:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset5_seq_915.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset5_spatial_915.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset5_seq_915_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset5_spatial_915_uce_adata.h5ad")
    elif dataset_num==6:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset6_seq_251.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset6_spatial_251.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset6_seq_251_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset6_spatial_251_uce_adata.h5ad")
    elif dataset_num==7:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset7_seq_118.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset7_spatial_118.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset7_seq_118_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset7_spatial_118_uce_adata.h5ad")
    elif dataset_num==8:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset8_seq_84.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset8_spatial_84.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset8_seq_84_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset8_spatial_84_uce_adata.h5ad")
    elif dataset_num==9:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset9_seq_76.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset9_spatial_76.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset9_seq_76_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset9_spatial_76_uce_adata.h5ad")
    elif dataset_num==10:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset10_seq_42.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset10_spatial_42.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset10_seq_42_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset10_spatial_42_uce_adata.h5ad")
    elif dataset_num==11:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset11_seq_347.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset11_spatial_347.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset11_seq_347_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset11_spatial_347_uce_adata.h5ad")
    elif dataset_num==12:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset12_seq_1000.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset12_spatial_1000.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset12_seq_1000_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset12_spatial_1000_uce_adata.h5ad")
    elif dataset_num==13:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset13_seq_154.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset13_spatial_154.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset13_seq_154_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset13_spatial_154_uce_adata.h5ad")
    elif dataset_num==14:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset14_seq_981.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset14_spatial_981.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset14_seq_981_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset14_spatial_981_uce_adata.h5ad")
    elif dataset_num==15:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset15_seq_141.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset15_spatial_141.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset15_seq_141_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset15_spatial_141_uce_adata.h5ad")
    elif dataset_num==16:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset16_seq_118.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset16_spatial_118.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset16_seq_118_uce_adata.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-uce-33/dataset16_spatial_118_uce_adata.h5ad")
    else:
        print('Add your datasets')
    
    return sc_ori, st_ori, sc_embed, st_embed 


def datasets_scGPT(dataset_num=2):
    '''
    sc data: obs: 'n_genes'(cells); var: 'n_cells'(genes)
    Spatial data: obs: 'x_coord', 'y_coord', 'n_genes'(number); var: 'n_cells'(genes)
    embed data: 33 --> ffeature = 512 scGPT / 1280 UCE
    '''
    if dataset_num==1:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset1_seq_246.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset1_spatial_246.h5ad")
        st_ori.obs['x_coord'] = st_ori.obs['x']
        st_ori.obs['y_coord'] = st_ori.obs['y']
        del st_ori.obs['x']
        del st_ori.obs['y']
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data1embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data1embed_st.h5ad")
    elif dataset_num==2:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset2_seq_33.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset2_spatial_33.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data2embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data2embed_st.h5ad")
    elif dataset_num==3:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset3_seq_42.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset3_spatial_42.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data3embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data3embed_st.h5ad")
    elif dataset_num==4:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset4_seq_1000.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset4_spatial_1000.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data4embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data4embed_st.h5ad")
    elif dataset_num==5:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset5_seq_915.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset5_spatial_915.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data5embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data5embed_st.h5ad")
    elif dataset_num==6:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset6_seq_251.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset6_spatial_251.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data6embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data6embed_st.h5ad")
    elif dataset_num==7:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset7_seq_118.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset7_spatial_118.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data7embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data7embed_st.h5ad")
    elif dataset_num==8:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset8_seq_84.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset8_spatial_84.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data8embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data8embed_st.h5ad")
    elif dataset_num==9:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset9_seq_76.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset9_spatial_76.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data9embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data9embed_st.h5ad")
    elif dataset_num==10:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset10_seq_42.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset10_spatial_42.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data10embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data10embed_st.h5ad")
    elif dataset_num==11:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset11_seq_347.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset11_spatial_347.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data11embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data11embed_st.h5ad")
    elif dataset_num==12:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset12_seq_1000.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset12_spatial_1000.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data12embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data12embed_st.h5ad")
    elif dataset_num==13:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset13_seq_154.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset13_spatial_154.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data13embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data13embed_st.h5ad")
    elif dataset_num==14:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset14_seq_981.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset14_spatial_981.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data14embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data14embed_st.h5ad")
    elif dataset_num==15:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset15_seq_141.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset15_spatial_141.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data15embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data15embed_st.h5ad")
    elif dataset_num==16:
        sc_ori = sc.read_h5ad("stDiff_Datasets/stDiff-sc/dataset16_seq_118.h5ad")
        st_ori = sc.read_h5ad("stDiff_Datasets/stDiff-st/dataset16_spatial_118.h5ad")
        # embedding
        sc_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data16embed_sc.h5ad") 
        st_embed = sc.read_h5ad("stDiff_Datasets/embed-scgpt/data16embed_st.h5ad")
    else:
        print('Add your datasets')
    
    # sc.pp.normalize_total(sc_ori, target_sum=100)
    # sc.pp.normalize_total(st_ori, target_sum=100)
    # sc.pp.log1p(sc_ori)
    # sc.pp.log1p(st_ori)
    return sc_ori, st_ori, sc_embed, st_embed 


def draw_data(target_gene, sc_ori, st_ori, Imp_Genes, args):
    
    ## predict/true data correlation
    x = st_ori[:, target_gene].X.reshape(-1)
    plt.figure(figsize=(12, 6))
    plt.scatter(x, Imp_Genes[target_gene], color='skyblue', label='Predicted vs True')
    plt.title("Scatter Plot of Predicted vs True Gene Expression {}".format(target_gene))
    plt.xlabel("Ture Gene Expression")
    plt.ylabel("Predict Gene Expression")
    path = os.path.join(args.dir, 'Scatter_data_{}_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)
    # plt.close()
    
    ## predict/true data
    plt.figure(figsize=(12, 6))
    # sns.histplot(data=original, bins=20, kde=True, element="step", fill=True)
    sns.kdeplot(x, shade=True, color='skyblue')
    # z = st.zscore(x, axis=0)
    # sns.kdeplot(z, shade=True, color='skyblue')
    plt.title("Gene Expression Density Distribution {}".format(target_gene))
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.xlim(-10, 40)
    # plt.legend(lineage[target_gene])
    path = os.path.join(args.dir, 'data_{}_true_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)
    # plt.close()

    plt.figure(figsize=(12, 6))
    # sns.histplot(data=original, bins=20, kde=True, element="step", fill=True)
    sns.kdeplot(Imp_Genes[target_gene], shade=True, color='skyblue')
    # z = st.zscore(Imp_Genes[target_gene], axis=0)
    # sns.kdeplot(z, shade=True, color='skyblue')
    plt.title("Gene Expression Density Distribution of Lineage {}".format(target_gene))
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.xlim(-10, 40)
    # plt.legend(lineage[i])
    path = os.path.join(args.dir, 'data_{}_predict_gene_{}.png'.format(args.dataset_num, target_gene))
    plt.savefig(path)
    # plt.close()
    
    
    ## predict voxel image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_facecolor('black')
    cmap = st_ori[:,target_gene].X
    cmap[cmap > np.percentile(cmap,99)] = np.percentile(cmap,99) # 所有大于其99百分位数的值替换为99百分位数的值，以使得颜色映射更加均匀
    ax1.scatter(st_ori.obs['x_coord'], st_ori.obs['y_coord'],s=1,c=cmap) # 每个点代表一个细胞，大小s为1，X Y 表示细胞的空间位置，cmap表示点的颜色i.e.该细胞的度量值
    ax1.axis('off')
    ax1.set_title('Measured ' + target_gene, fontsize = 16, color='white')
    
    # ax2.set_facecolor('black')
    cmap = Imp_Genes[target_gene]
    cmap[cmap > np.percentile(cmap,99)] = np.percentile(cmap,99)
    ax2.scatter(st_ori.obs['x_coord'], st_ori.obs['y_coord'],s=1,c=cmap)
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
    # sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
    # sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    # sc.tl.umap(adata, min_dist=0.3, n_components=2)
    # sc.pl.embedding(adata, basis='umap',color='batch',color_map='tab20')
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    # pdb.set_trace()
    sc.pl.umap(adata, color='label',palette=color_map, ax=ax)
    
    plt.title("UMAP of Data {}".format(args.dataset_num), fontsize=18)
    plt.xlabel("UMAP_1", fontsize=16)
    plt.ylabel("UMAP_2", fontsize=16)
    path = os.path.join(args.dir, 'Umap_data_{}.png'.format(args.dataset_num))
    plt.legend(prop={'size': 16}, bbox_to_anchor=(1.0, 0.5), loc='center left', frameon=False)  # 显示图例并设置大小
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(path)
    plt.close()
    
    

         
if __name__ == '__main__':
    #               1    2   3    4     5    6    7   8    9  10   11    12   13   14   15   16
    shared_genes = [246, 33, 42, 1000, 915, 251, 118, 84, 76, 42, 347, 1000, 154, 981, 141, 118]  
    datasetall = [[index, value] for index, value in enumerate(shared_genes)]
    
    # data_num = [1,2,3,6,7,8,9,10,11,13]
    # data_num = [15,16,14,12,4,5]
    # data_num = [9,10,11,13]

    # data_num = [ 1] #2, 3,
    # data_num = [4]#6, 13, 
    # data_num = [9] #5
    # data_num = [7, 10, 9] #8, 

    # data_num = [16, 15] 
    # data_num = [11] 
    # data_num = [12] 
    # data_num = [14] 

    
    # data_num = [1, 2,4,5,8,10,11, 12,14, 15,16]
    
    # datasetall = []
    # for i in data_num:
    #     datasetall.append([i, shared_genes[i-1]])
    
    for index, num in datasetall:
        parser = argparse.ArgumentParser(description='Gene Alignment Projection Model')
        parser.add_argument('--dataset_num', default=index+1, type=int, help='the number of the datasets')
        # parser.add_argument('--dataset_num', default=index, type=int, help='the number of the datasets')
        parser.add_argument('--test_gene_num', default=num, type=int, help='the number of target genes')
        # parser.add_argument('--test_gene_num', default=2, type=int, help='the number of target genes')
        parser.add_argument('--latent_dim', default=512, type=int, help='latent alignment dimention')
        parser.add_argument('--encode_dim', default=4, type=int, help='encode dimention')
        # parser.add_argument('--dir', default='./output_embed/', type=str, help='working folder where all files will be saved.')
        parser.add_argument('--dir', default='./output/', type=str, help='working folder where all files will be saved.')
        
        parser.add_argument('--proj', default=1, type=int, help='impute weight')
        parser.add_argument('--norm', default=1, type=int, help='impute weight')
        parser.add_argument('--lr', default=0.1, type=int, help='learning rate')
        parser.add_argument('--wd', default=1e-8, type=int, help='weight decay')
        parser.add_argument('--epoch', default=200, type=int, help='training epoch')
        args = parser.parse_args()
        print(args)
        
        # Dataset prepare
        # _, _, sc_embed_scgpt, st_embed_scgpt = datasets_scGPT(dataset_num=args.dataset_num)
        _, _, sc_embed_uce, st_embed_uce = datasets_UCE_4(dataset_num=args.dataset_num)
        sc_ori, st_ori, sc_embed_uce2, st_embed_uce2 = datasets_UCE_4(dataset_num=args.dataset_num)
        
        
        Gene_set = sc_ori.var_names[:args.test_gene_num]
        # pdb.set_trace()
        # args.latent_dim = sc_ori.shape[0]

        # Prediction results
        Correlations = pd.Series(index = Gene_set) 
        PCC = pd.Series(index = Gene_set) 
        SSIM = pd.Series(index = Gene_set)
        JS = pd.Series(index = Gene_set) 
        RMSE = pd.Series(index = Gene_set) 
        All_Imp_Genes = pd.DataFrame(np.zeros((st_ori.shape[0], args.test_gene_num)),columns=Gene_set)
        Dic_data =  pd.DataFrame({'column': [np.zeros((sc_ori.shape[0], st_ori.shape[0])) for _ in Gene_set]}, index=Gene_set)
        # pdb.set_trace()
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # leave one out prediction
        for i in Gene_set:
            Imp_Genes, Dic = GAP(st_ori[:, ~st_ori.var_names.isin([i])], st_ori[:, st_ori.var_names.isin([i])], sc_ori, 
                            st_embed_uce2[:, ~st_embed_uce2.var_names.isin([i])], sc_embed_uce2, st_embed_uce[:, ~st_embed_uce.var_names.isin([i])], sc_embed_uce, 
                            args, genes_to_predict=[i])
            # pdb.set_trace()
            st_ori[:, i].X = scaler.fit_transform(st_ori[:, i].X)
            
            Correlations[i] = st.spearmanr(st_ori[:, i].X, Imp_Genes[i])[0]
            x = st_ori[:, i].X.reshape(-1)
            PCC[i] =  np.corrcoef(x, Imp_Genes[i])[0, 1]
            data_range = max(x.max(), Imp_Genes[i].max()) - min(x.min(), Imp_Genes[i].min())
            SSIM[i] = ssim(x, Imp_Genes[i], data_range=data_range)
            m = 0.5 * (x + Imp_Genes[i])
            JS[i] = 0.5 * (entropy(x, m) + entropy(Imp_Genes[i], m))
            RMSE[i] = np.sqrt(np.mean((x - Imp_Genes[i]) ** 2))
            All_Imp_Genes[i] = Imp_Genes[i]
            Dic_data.at[i, 'column'] = Dic

        # save results
        for item in Correlations.index:
            print(str(item))
        for item in Correlations.values:
            print(str(item))
        print('Correlations: \n', Correlations)
        print('Correlations: ', Correlations.values)
        print('PCC: ', PCC.values)
        print('SSIM: ', SSIM.values)
        print('JS: ', JS.values)
        print('RMSE: ', RMSE.values)
        path = os.path.join(args.dir, 'gap+_data_{}_gene_{}_ldim_{}.txt'.format(args.dataset_num, args.test_gene_num, args.latent_dim))
        with open(path, 'w') as f:
            for item in Correlations.index:
                f.write(str(item) + '\n')
            f.write('\n')
            for item in Correlations.values:
                f.write(str(item) + '\n')
            f.write('\n'+'Spearman_data_'+ str(args.dataset_num)  + '\n'+ str(Correlations) + '\n\n')
            f.write('Spearman_data_'+ str(args.dataset_num)  + '\n'+ str(Correlations.values) + '\n\n')
            f.write('PCC_data_'+ str(args.dataset_num)  + '\n'+ str(PCC.values) + '\n\n')
            f.write('SSIM_data_'+ str(args.dataset_num)  + '\n'+ str(SSIM.values) + '\n\n')
            f.write('JS_data_'+ str(args.dataset_num)  + '\n'+ str(JS.values) + '\n\n')
            f.write('RMSE_data_'+ str(args.dataset_num)  + '\n'+ str(RMSE.values) + '\n\n\n\n')
            
            f.write('Average_Spearman_data_'+ str(args.dataset_num)  + '\n'+ str(np.mean(Correlations.values)) + '\n\n')
            f.write('Average_PCC_data_'+ str(args.dataset_num)  + '\n'+ str(np.mean(PCC.values)) + '\n\n')
            f.write('Average_SSIM_data_'+ str(args.dataset_num)  + '\n'+ str(np.mean(SSIM.values)) + '\n\n')
            f.write('Average_JS_data_'+ str(args.dataset_num)  + '\n'+ str(np.mean(JS.values)) + '\n\n')
            f.write('Average_RMSE_data_'+ str(args.dataset_num)  + '\n'+ str(np.mean(RMSE.values)) + '\n\n')
        
        # save imputation results
        All_Imp_Genes.to_csv(os.path.join(args.dir, 'gap+_all_imputation_data_{}.txt'.format(args.dataset_num)), sep='\t', index=True)
        # save dictionary results
        with open(os.path.join(args.dir, 'gap+_all_Dic_data_{}.txt'.format(args.dataset_num)), 'wb') as f:
            pickle.dump(Dic_data, f)

        # draw results
        gene_num = [0]
        for i in gene_num:
            # D = Dic_data.at[i,'column']
            draw_data(Gene_set[i], sc_ori, st_ori, All_Imp_Genes, args)
        
        # Unknown Gene Prediction
        # new_genes = ['Cux2','Tmem215','Pvrl3','Wfs1','Adam33','Rsto1','Tesc','Tox','Foxp2','Tle4']
        # Imp_New_Genes = GAP(st_ori, st_ori, st_ori, sc_ori, st_embed, sc_embed, 
        #                     args, genes_to_predict=new_genes)
        # draw_data(i, st_ori, Imp_New_Genes)
        # for i in [0,2,4,6,8]:
        #     draw_data(new_genes[i], st_ori, Imp_New_Genes)
    
    # conda activate gap        
    # CUDA_VISIBLE_DEVICES='0' python main_embed.py
    # CUDA_VISIBLE_DEVICES='1' python main_embed.py
    
# fuser -k /dev/nvidia0
# sudo scp -P 2350 -r liqing@projgw.cse.cuhk.edu.hk:/data/liqing/projects/SpaGE-master/gap.py /home/casp16/GAP
# sudo scp -P 2350 -r liqing@projgw.cse.cuhk.edu.hk:/data/liqing/projects/SpaGE-master/gap_model.py /home/casp16/GAP


