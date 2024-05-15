import torch
import torch.nn as nn
import torch.nn.functional as F


class DicLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DicLayer, self).__init__()
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x):
        W = torch.abs(self.W)
        return torch.matmul(x, W)

class AxisLayer(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(AxisLayer, self).__init__()
        self.M = nn.Parameter(torch.abs(torch.randn(output_dim, input_dim)))

    def forward(self, x):
        row_sums = self.M.sum(dim=1)
        M = self.M / row_sums.view(-1, 1)
        result = torch.matmul(torch.relu(M), x)

        return result, torch.relu(M)
    
class DicModel(nn.Module):
    def __init__(self, sc_input, st_input, latent_dim):
        super(DicModel, self).__init__()
        in_dim = sc_input.shape[1]
        cell_dim = sc_input.shape[0]
        voxel_dim = st_input.shape[0]
        
        self.dic_layer = DicLayer(in_dim, latent_dim)
        self.fc = nn.Linear(latent_dim, in_dim)
        
        self.cell2voxel = AxisLayer(voxel_dim, cell_dim)
        
        self.fc1 = nn.Linear(in_dim, latent_dim)        
        self.fc2 = nn.Linear(latent_dim, in_dim)  

    def forward(self, x, y):
        x = self.dic_layer(x)
        voxel = self.dic_layer(y)
        
        cell2voxel, M = self.cell2voxel(x)
        
        sc_out = self.fc(x)
        sp_out = self.fc(voxel)
        
        return sc_out, sp_out, cell2voxel, voxel, M 
    

