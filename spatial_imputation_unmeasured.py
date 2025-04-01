import argparse
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from tqdm import tqdm
from src.model_fat import *
from src.dataset import *
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import pickle
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def mmd_loss(x, y):
    x = F.softmax(x)
    y = F.softmax(y)
    p_joint = torch.exp(-torch.norm(x - y, dim=1))
    p_x = torch.mean(p_joint, dim=0)
    p_y = torch.mean(p_joint, dim=0) 
    mi = torch.mean(torch.log(p_joint / (p_x.unsqueeze(0) * p_y.unsqueeze(0)))) 
    return mi

def FAT(st_train, sc_train, st_test, sc_test, args):
    Y_train = sc_test

    # unmeasured statial genes expression to be predicted
    Imp_Genes = pd.DataFrame(np.zeros(st_test.shape),columns=st_test.var_names)
    Dic = np.zeros((st_train.shape[0],sc_train.shape[0]))
    
    # ori data
    sc_ori_scaled = pd.DataFrame(sc_train.X, columns=sc_train.var_names)
    st_ori_scaled = pd.DataFrame(st_train.X, columns=st_train.var_names) # index=axis_index,

    # shared ori data
    sc_intsec_scaled = sc_ori_scaled[np.intersect1d(sc_ori_scaled.columns,st_ori_scaled.columns)]
    st_intsec_scaled = st_ori_scaled[sc_intsec_scaled.columns]

    scaler = MinMaxScaler(feature_range=(0, 1))

    sc_input = torch.tensor(scaler.fit_transform(sc_intsec_scaled.values)).to(torch.float32).to(device) 
    st_input = torch.tensor(scaler.fit_transform(st_intsec_scaled.values)).to(torch.float32).to(device)
    
    model  = DicModel(ori_dim=sc_ori_scaled.shape[1],sc_input=sc_input, st_input=st_input, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 

    for epoch in tqdm(range(args.epoch)): 
        x_enc, y_enc, x_dic, y_dic, cell2voxel, W, D = model(sc_input, st_input)
        
        l_align =  - mmd_loss(sc_input[:,:st_ori_scaled.shape[1]], x_enc) \

        l_proj = -F.cosine_similarity(cell2voxel, y_dic, dim=0).mean() \
            -F.cosine_similarity(torch.matmul(W, sc_input[:,:st_ori_scaled.shape[1]]), st_input[:,:st_ori_scaled.shape[1]], dim=0).mean()\

        loss =   args.proj * l_proj  + l_align 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            D = W.double().cpu().detach().numpy()

    Imp_Genes[:]= np.dot(D, scaler.fit_transform(Y_train.X))
    Dic = D
    
    return Imp_Genes, Dic  

         
if __name__ == '__main__':
    shared_genes = [33, 42, 915, 251, 118, 76, 42, 347, 1000, 154, 141, 118] 
    datasetall = [[index, value] for index, value in enumerate(shared_genes)]
    
    for index, num in datasetall:
        parser = argparse.ArgumentParser(description='FAT Model')
        parser.add_argument('--dataset_num', default=index+1, type=int, help='the number of the datasets')
        parser.add_argument('--test_gene_num', default=num, type=int, help='the number of target genes')
        parser.add_argument('--latent_dim', default=20+num, type=int, help='latent alignment dimention')
        parser.add_argument('--encode_dim', default=10, type=int, help='encode dimention')
        parser.add_argument('--dir', default='./output/', type=str, help='working folder where all files will be saved.')
        
        parser.add_argument('--proj', default=1, type=int, help='impute weight')
        parser.add_argument('--norm', default=1, type=int, help='impute weight')
        parser.add_argument('--lr', default=0.1, type=int, help='learning rate')
        parser.add_argument('--wd', default=1e-8, type=int, help='weight decay')
        parser.add_argument('--epoch', default=200, type=int, help='training epoch')
        parser.add_argument("--n_splits", type=int, default=5, help='cross-validation sets number')
        args = parser.parse_args()
        print(args)
        
        # Dataset prepare
        st_train, sc_train, st_test, sc_test = datasets_82(dataset_num=args.dataset_num, part=0)
        sc_ori = ad.concat([sc_train, sc_test], axis=1, merge="same")
        st_ori = ad.concat([st_train, st_test], axis=1, merge="same")
        Gene_set = st_ori.var_names

        # Prediction results
        Correlations = pd.Series(index = Gene_set) 
        PCC = pd.Series(index = Gene_set) 
        SSIM = pd.Series(index = Gene_set)
        JS = pd.Series(index = Gene_set) 
        RMSE = pd.Series(index = Gene_set) 
        All_Imp_Genes = pd.DataFrame(np.zeros((st_ori.shape[0], args.test_gene_num)),columns=Gene_set)
        Dic_data =  pd.DataFrame({'column': [np.zeros((sc_ori.shape[0], st_ori.shape[0])) for _ in Gene_set]}, index=Gene_set)
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Training process
        for part in range(args.n_splits):
            st_train, sc_train, st_test, sc_test = datasets_82(dataset_num=args.dataset_num, part=part)
            sc_ori = ad.concat([sc_train, sc_test], axis=1, merge="same")
            st_ori = ad.concat([st_train, st_test], axis=1, merge="same")
            Imp_Genes, Dic = FAT(st_train, sc_train, st_test, sc_test, args)
            for i in st_test.var_names:
                st_test[:, i].X = scaler.fit_transform(st_test[:, i].X)
                Correlations[i] = st.spearmanr(st_test[:, i].X, Imp_Genes[i])[0]
                x = st_test[:, i].X.reshape(-1)
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
        path = os.path.join(args.dir, 'gap_data_{}_gene_{}_ldim_{}.txt'.format(args.dataset_num, args.test_gene_num, args.latent_dim))
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
        
        # draw results
        gene_num = [0]
        for i in gene_num:
            draw_data(Gene_set[i], sc_ori, st_ori, All_Imp_Genes, args)
        
# conda activate fat      
# CUDA_VISIBLE_DEVICES='0' python spatial_imputation_unmeaured.py
