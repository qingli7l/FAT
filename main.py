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
from model_embed import *
from dataset import *
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
    # norm ori data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # sc_ori_scaled = pd.DataFrame(data=scaler.fit_transform(sc_ori.X), columns=sc_ori.var_names)
    # st_ori_scaled = pd.DataFrame(data=scaler.fit_transform(st_ori.X), columns=st_ori.var_names) # index=axis_index,
    
    # shared ori data
    sc_intsec_scaled = sc_ori_scaled[np.intersect1d(sc_ori_scaled.columns,st_ori_scaled.columns)]
    st_ori_scaled = st_ori_scaled[sc_intsec_scaled.columns]

    # 4 level uce embedding data
    st_embed_uce = pd.DataFrame(st_embed_uce.obsm["X_uce"])
    sc_embed_uce = pd.DataFrame(sc_embed_uce.obsm["X_uce"])
    # 33 level uce embedding data
    st_embed_uce2 = pd.DataFrame(st_embed_uce2.obsm["X_uce"])
    sc_embed_uce2 = pd.DataFrame(sc_embed_uce2.obsm["X_uce"])
    # pdb.set_trace()
    # nan_ratio = np.isnan(sc_embed_uce2.values).sum() / np.prod(sc_embed_uce2.values.shape)
    # print(f"NaN ratio in the data: {nan_ratio * 100:.2f}%")
    
    # concate ori and embedding data
    st_ori_concate = pd.concat([st_ori_scaled,st_embed_uce,st_embed_uce2], axis=1)
    sc_ori_concate = pd.concat([sc_intsec_scaled,sc_embed_uce,sc_embed_uce2], axis=1)
    # st_ori_concate = pd.concat([st_ori_scaled,st_embed_uce], axis=1)
    # sc_ori_concate = pd.concat([sc_intsec_scaled,sc_embed_uce], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # GAP
    sc_input = torch.tensor(scaler.fit_transform(sc_ori_concate.values)).to(torch.float32).to(device) 
    st_input = torch.tensor(scaler.fit_transform(st_ori_concate.values)).to(torch.float32).to(device)
    
    model  = DicModel(ori_dim=st_ori_scaled.shape[1],embed1_dim=st_embed_uce.shape[1],sc_input=sc_input, st_input=st_input, latent_dim=args.latent_dim, encode_dim=args.encode_dim).to(device)
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
  
    

         
if __name__ == '__main__':
    shared_genes = [246, 33, 42, 1000, 915, 251, 118, 84, 76, 42, 347, 1000, 154, 981, 141, 118]  
    datasetall = [[index, value] for index, value in enumerate(shared_genes)]

    for index, num in datasetall:
        parser = argparse.ArgumentParser(description='single-cell Gene Alignment Projection Model')
        parser.add_argument('--dataset_num', default=index+1, type=int, help='the number of the datasets')
        parser.add_argument('--test_gene_num', default=num, type=int, help='the number of target genes')
        parser.add_argument('--latent_dim', default=512, type=int, help='latent alignment dimention')
        parser.add_argument('--encode_dim', default=4, type=int, help='encode dimention')
        parser.add_argument('--dir', default='./output_embed4+33/', type=str, help='working folder where all files will be saved.')
        
        parser.add_argument('--proj', default=1, type=int, help='impute weight')
        parser.add_argument('--norm', default=1, type=int, help='impute weight')
        parser.add_argument('--lr', default=0.1, type=int, help='learning rate')
        parser.add_argument('--wd', default=1e-8, type=int, help='weight decay')
        parser.add_argument('--epoch', default=200, type=int, help='training epoch')
        args = parser.parse_args()
        print(args)
        
        # Dataset prepare
        _, _, sc_embed_uce, st_embed_uce = datasets_UCE_4(dataset_num=args.dataset_num)
        sc_ori, st_ori, sc_embed_uce2, st_embed_uce2 = datasets_UCE_33(dataset_num=args.dataset_num)
        
        
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
        # All_Imp_Genes.to_csv(os.path.join(args.dir, 'gap+_all_imputation_data_{}.txt'.format(args.dataset_num)), sep='\t', index=True)
        # save dictionary results
        # with open(os.path.join(args.dir, 'gap+_all_Dic_data_{}.txt'.format(args.dataset_num)), 'wb') as f:
        #     pickle.dump(Dic_data, f)

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
