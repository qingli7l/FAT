import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from tqdm import tqdm
from src.model_fat import *
from src.dataset import *
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def mmd_loss(x, y):
    x = F.softmax(x, dim=1)
    y = F.softmax(y, dim=1)
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
    sc_ori_scaled = pd.DataFrame(sc_train.X.toarray(), columns=sc_train.var_names)
    st_ori_scaled = pd.DataFrame(st_train.X.toarray(), columns=st_train.var_names) 

    # shared ori data
    sc_intsec_scaled = sc_ori_scaled[np.intersect1d(sc_ori_scaled.columns,st_ori_scaled.columns)]
    st_intsec_scaled = st_ori_scaled[sc_intsec_scaled.columns]

    scaler = MinMaxScaler(feature_range=(0, 1))

    # FAT
    sc_input = torch.tensor(scaler.fit_transform(sc_intsec_scaled.values)).to(torch.float32).to(device) 
    st_input = torch.tensor(scaler.fit_transform(st_intsec_scaled.values)).to(torch.float32).to(device)
    
    model  = DicModel(ori_dim=sc_ori_scaled.shape[1],sc_input=sc_input, st_input=st_input, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 

    # for epoch in tqdm(range(args.epoch)): 
    for epoch in range(args.epoch): 
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

    Imp_Genes[:]= np.dot(D, scaler.fit_transform(Y_train.X.toarray()))
    Dic = D

    return Imp_Genes, Dic  
  
    

         
if __name__ == '__main__':
    DATASET = 'zhuang'

    batch_num = 1 
    # batch_num = 2
    # batch_num = 3 
    # batch_num = 4

    sc_ori = sc.read_h5ad("src/human_data/dataset_zhuang/sc.h5ad")
    if batch_num == 1:
        st_ori = sc.read_h5ad("src/human_data/dataset_zhuang/st1.h5ad") 
    if batch_num == 2:
        st_ori = sc.read_h5ad("src/human_data/dataset_zhuang/st2.h5ad")
    if batch_num == 3:
        st_ori = sc.read_h5ad("src/human_data/dataset_zhuang/st3.h5ad")
    if batch_num == 4:
        st_ori = sc.read_h5ad("src/human_data/dataset_zhuang/st4.h5ad")
    sc_ori.obs['batch']= sc_ori.obs['cluster'] 
    st_ori.obs['batch']= st_ori.obs['brain_section_label_y']
    print('ori sc shape: ',sc_ori.X.shape, 'st shape: ',st_ori.X.shape)
    
    # filter lowely expressed genes
    sc_ori.obs['Genes_count'] = np.sum(sc_ori.X > 0, axis=1) 
    sc_ori = sc_ori[sc_ori.obs['Genes_count'] >=10] 
    st_ori.obs['Genes_count'] = np.sum(st_ori.X > 0, axis=1) 
    st_ori = st_ori[st_ori.obs['Genes_count'] >=10]
    
    shared_gene = pd.read_csv('src/batch_shared_genes.txt', sep='\t', header=None)[1].tolist()
    shared_gene = [gene.strip() for gene in shared_gene]
    shared_gene = [gene for gene in shared_gene if gene in sc_ori.var_names]

    sc_ori = sc_ori[:,shared_gene]
    st_ori = st_ori[:,shared_gene]
    
    sc_ori = sc_ori[:1000,:]
    st_ori = st_ori[:500,:]
    
    print('filtered sc shape: ',sc_ori.X.shape, 'st shape: ',st_ori.X.shape)
    
    # sample 25 genes of each group in sparcity boundaries = [0, 0.75, 0.9, 0.95, 1]
    target_genes = stratified_sample_genes_by_sparsity(st_ori, seed=11)
    
    imput_gene = shared_gene[2]
    target_genes.append(imput_gene)
    num = len(target_genes)
    parser = argparse.ArgumentParser(description='FAT Model')
    parser.add_argument('--dataset_num', default=DATASET, type=str, help='the number of target genes')
    parser.add_argument('--test_gene_num', default=num, type=int, help='the number of target genes')
    parser.add_argument('--latent_dim', default=20+num, type=int, help='latent alignment dimention')
    parser.add_argument('--encode_dim', default=10, type=int, help='encode dimention')
    parser.add_argument('--dir', default='./output/', type=str, help='working folder where all files will be saved.')
    
    parser.add_argument('--proj', default=1, type=int, help='impute weight')
    parser.add_argument('--norm', default=1, type=int, help='impute weight')
    parser.add_argument('--lr', default=0.1, type=int, help='learning rate')
    parser.add_argument('--wd', default=1e-8, type=int, help='weight decay')
    parser.add_argument('--epoch', default=20, type=int, help='training epoch')
    parser.add_argument("--n_splits", type=int, default=5, help='cross-validation sets number')
    
    parser.add_argument('--batch_num', default=batch_num, type=int, help='batch num')
    
    args = parser.parse_args()
    print(args)

    # Prediction results
    Correlations = pd.Series(index = target_genes) 
    PCC = pd.Series(index = target_genes) 
    SSIM = pd.Series(index = target_genes)
    JS = pd.Series(index = target_genes) 
    RMSE = pd.Series(index = target_genes) 
    MSE = pd.Series(index = target_genes) 
    MAE = pd.Series(index = target_genes) 
    COS = pd.Series(index = target_genes) 
    All_Imp_Genes = pd.DataFrame(np.zeros((st_ori.shape[0], args.test_gene_num)),columns=target_genes)
    Dic_data =  pd.DataFrame({'column': [np.zeros((sc_ori.shape[0], st_ori.shape[0])) for _ in target_genes]}, index=target_genes)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for i in tqdm(target_genes):
        Imp_Genes, Dic  = FAT(st_ori[:, ~st_ori.var_names.isin([i])], sc_ori[:, ~st_ori.var_names.isin([i])],
                        st_ori[:, st_ori.var_names.isin([i])], sc_ori[:, st_ori.var_names.isin([i])], args)
        
        x = scaler.fit_transform(st_ori[:, i].X.toarray()).reshape(-1)
        predict = np.array(Imp_Genes[i]).flatten()
        
        MSE[i] = mean_squared_error(x, predict)
        MAE[i] = mean_absolute_error(x, predict)
        COS[i] = 1 - cosine(x, predict)
        
        Correlations[i] = st.spearmanr(x, predict)[0]
        PCC[i] =  np.corrcoef(x, predict)[0, 1]
        data_range = max(x.max(), predict.max()) - min(x.min(), predict.min())
        SSIM[i] = ssim(x, predict, data_range=data_range)
        m = 0.5 * (x + predict)
        JS[i] = 0.5 * (entropy(x, m) + entropy(predict, m))
        RMSE[i] = np.sqrt(np.mean((x - predict) ** 2))
        All_Imp_Genes[i] = predict.reshape(-1,1)

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
    print('MAE: ', MAE.values)
    print('MSE: ', MSE.values)
    print('COS: ', COS.values)
    path = os.path.join(args.dir, 'fat_data_{}_gene_{}_ldim_{}_batch_{}.txt'.format(DATASET, args.test_gene_num, args.latent_dim, batch_num))
    with open(path, 'w') as f:
        for item in Correlations.index:
            f.write(str(item) + '\n')
        f.write('\n')
        for item in Correlations.values:
            f.write(str(item) + '\n')
        f.write('\n'+'Spearman_data_'+ str(DATASET)  + '\n'+ str(Correlations) + '\n\n')
        f.write('Spearman_data_'+ str(DATASET)  + '\n'+ str(Correlations.values) + '\n\n')
        f.write('PCC_data_'+ str(DATASET)  + '\n'+ str(PCC.values) + '\n\n')
        f.write('SSIM_data_'+ str(DATASET)  + '\n'+ str(SSIM.values) + '\n\n')
        f.write('JS_data_'+ str(DATASET)  + '\n'+ str(JS.values) + '\n\n')
        f.write('RMSE_data_'+ str(DATASET)  + '\n'+ str(RMSE.values) + '\n\n\n\n')
        f.write('MAE_data_'+ str(DATASET)  + '\n'+ str(MAE.values) + '\n\n\n\n')
        f.write('MSE_data_'+ str(DATASET)  + '\n'+ str(MSE.values) + '\n\n\n\n')
        f.write('COS_data_'+ str(DATASET)  + '\n'+ str(COS.values) + '\n\n\n\n')
        
        f.write('Average_Spearman_data_'+ str(DATASET)  + '\n'+ str(np.mean(Correlations.values)) + '\n\n')
        f.write('Average_PCC_data_'+ str(DATASET)  + '\n'+ str(np.mean(PCC.values)) + '\n\n')
        f.write('Average_SSIM_data_'+ str(DATASET)  + '\n'+ str(np.mean(SSIM.values)) + '\n\n')
        f.write('Average_JS_data_'+ str(DATASET)  + '\n'+ str(np.mean(JS.values)) + '\n\n')
        f.write('Average_RMSE_data_'+ str(DATASET)  + '\n'+ str(np.mean(RMSE.values)) + '\n\n')
        f.write('Average_MAE_data_'+ str(DATASET)  + '\n'+ str(np.mean(MAE.values)) + '\n\n')
        f.write('Average_MSE_data_'+ str(DATASET)  + '\n'+ str(np.mean(MSE.values)) + '\n\n')
        f.write('Average_COS_data_'+ str(DATASET)  + '\n'+ str(np.mean(COS.values)) + '\n\n')
    
    # draw results
    draw_data_batch(imput_gene, sc_ori, st_ori, All_Imp_Genes, DATASET, args)


# conda activate fat      
# CUDA_VISIBLE_DEVICES='0' python spatial_batch_effect.py
