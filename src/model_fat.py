import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from torch.nn.functional import softmax


class DicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DicLayer, self).__init__()
        self.D = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x):
        D = self.D
        return torch.matmul(x, D), D

class AxisLayer(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(AxisLayer, self).__init__()
        self.M = nn.Parameter(torch.abs(torch.randn(output_dim, input_dim)))

    def forward(self, x):
        M = torch.abs(self.M)
        result = torch.matmul(M, x)

        return result, M


class DicModel(nn.Module):
    def __init__(self,ori_dim, sc_input, st_input, latent_dim):
        super(DicModel, self).__init__()
        cell_dim = sc_input.shape[0]
        voxel_dim = st_input.shape[0]
        self.ori_dim = ori_dim
        self.embed_dim = ori_dim
        self.encode_dim = ori_dim
        # pdb.set_trace()

        self.encoder = nn.Sequential(
            nn.Linear(self.embed_dim, 2*self.encode_dim),
            nn.Sigmoid(),
            nn.Linear(2*self.encode_dim, self.encode_dim),
            nn.Sigmoid()
        )
        
        self.dic_layer = DicLayer(ori_dim+self.encode_dim, latent_dim)
        
        self.cell2voxel = AxisLayer(voxel_dim, cell_dim)


    def forward(self, x, y):
        x_enc = self.encoder(-x)
        x = torch.cat([x, x_enc], dim=1)
        
        y_enc = self.encoder(-y)
        y = torch.cat([y, y_enc], dim=1)
        
        x_dic, D = self.dic_layer(x)
        y_dic, D = self.dic_layer(y)
        
        cell2voxel, W = self.cell2voxel(x_dic)
        
        return x_enc, y_enc, x_dic, y_dic, cell2voxel, W, D
  
  

