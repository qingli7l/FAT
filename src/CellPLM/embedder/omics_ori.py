import torch
from torch import nn
import torch.nn.functional as F
from CellPLM.utils.pe import select_pe_encoder
from CellPLM.utils import create_norm, create_activation
import numpy as np
from CellPLM.utils.sparse import sparse_normalize, sparse_tpm

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  # No activation or dropout after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
def make_fc_layers(linear_cfg, input_channels, output_channels, output_use_norm=False, ksize=1, pad=0):
    fc_layers = []
    c_in = input_channels
    for k in range(0, linear_cfg.__len__()):
        fc_layers.extend([
            nn.Conv1d(c_in, linear_cfg[k], kernel_size=ksize, padding=pad, bias=False),
            nn.BatchNorm1d(linear_cfg[k], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ])
        c_in = linear_cfg[k]
    if output_use_norm:
        fc_layers.append(nn.Conv1d(c_in, output_channels, kernel_size=ksize, padding=pad, bias=False)),
        fc_layers.append(nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01)),
        fc_layers.append(nn.ReLU()),
    else:
        fc_layers.append(nn.Conv1d(c_in, output_channels, kernel_size=ksize, padding=pad, bias=True))
    return nn.Sequential(*fc_layers)

class OmicsEmbedderWithAttention(nn.Module):
    def __init__(self, pretrained_gene_list, num_hid, gene_emb=None, fix_embedding=False, num_heads=8):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = dict(zip(pretrained_gene_list, list(range(len(pretrained_gene_list)))))
        self.num_hid = num_hid
        self.num_heads = num_heads

        if gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
        else:
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            if fix_embedding:
                self.emb.requires_grad = False

        # self.MLP = MLP(1, [64, 32], num_hid)
        # self.MLP = MLP(423, [512], num_hid)
        # self.MLP = nn.Linear(423, num_hid)

        # self.MLP = make_fc_layers([512], 1,  num_hid)
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(num_hid, num_heads,batch_first=True)
        
        # Layer normalization and feedforward layer
        # self.layer_norm1 = nn.LayerNorm(num_hid)
        # self.layer_norm2 = nn.LayerNorm(num_hid)
        self.feedforward = nn.Sequential(
            nn.Linear(num_hid, num_hid * 4),
            nn.ReLU(),
            nn.Linear(num_hid * 4, num_hid)
        )

    def forward(self, x_dict, input_gene_list=None):
        if 'masked_x_seq' in x_dict:
            x = x_dict['masked_x_seq']
        else:
            x = x_dict['x_seq']

        if 'dropout' in x_dict:
            indices = x._indices().t()
            values = x._values()
            values = values.float()
            values = torch.distributions.binomial.Binomial(values, x_dict['dropout']).sample()
            x = torch.sparse.FloatTensor(indices.t(), values, x.shape)

        x = torch.log1p(x)

        if input_gene_list is not None:
            # filterred_gene_list = [gene for gene, keep in zip(input_gene_list, x_dict['gene_mask']) if keep]
            # gene_idx = torch.tensor([self.gene_index[o] for o in filterred_gene_list if o in self.gene_index]).long()
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()           
            x_dict['input_gene_mask'] = gene_idx
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError('The input gene size is not the same as the pretrained gene list. Please provide the input gene list.')
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)

        feat = F.embedding(gene_idx, self.emb)
        # feat = torch.sparse.mm(x, feat)/len(gene_idx)

        # adding attention
        x = x.to_dense().unsqueeze(-1)
        
        feat = x * feat
        query = feat.sum(1,keepdim=True)
        
        attn_output, _ = self.attention(query, feat, feat)
        feat = query + attn_output
        ff_output = self.feedforward(feat)
        feat = feat + ff_output
        feat = feat.squeeze(1)

        '''
        x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)
        x = x.to_dense()
        # import pdb; pdb.set_trace()
        feat = self.MLP(x) 
        feat = feat.permute(0, 2, 1)
        # Apply attention
        # feat = feat.unsqueeze(0)  # Add sequence dimension
        attn_output, _ = self.attention(query, feat, feat)
        # attn_output = attn_output.squeeze(1)  # Remove sequence dimension

        query = query + attn_output
        # feat = feat + attn_output
        ff_output = self.feedforward(query)
        query = query + ff_output
        query = query.squeeze(1)
        '''
        return feat
    
class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hid, gene_emb=None, fix_embedding=False):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = dict(zip(pretrained_gene_list, list(range(len(pretrained_gene_list)))))
        self.num_hid = num_hid

        if gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
        else:
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            if fix_embedding:
                self.emb.requires_grad = False

    def forward(self, x_dict, input_gene_list=None):
        if 'masked_x_seq' in x_dict:
            x = x_dict['masked_x_seq']
        else:
            x = x_dict['x_seq']

        if 'dropout' in x_dict:
            indices = x._indices().t()
            values = x._values()
            temp = values.sum()
            values = values.float()
            values = torch.distributions.binomial.Binomial(values, x_dict['dropout']).sample()
            x = torch.sparse.FloatTensor(indices.t(), values, x.shape)

        x = torch.log1p(x)
        # x = sparse_tpm(x)
        if input_gene_list is not None:
            # filterred_gene_list = [gene for gene, keep in zip(input_gene_list, x_dict['gene_mask']) if keep]
            # gene_idx = torch.tensor([self.gene_index[o] for o in filterred_gene_list if o in self.gene_index]).long() 
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()           
            x_dict['input_gene_mask'] = gene_idx
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError('The input gene size is not the same as the pretrained gene list. Please provide the input gene list.')
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)

        feat = F.embedding(gene_idx, self.emb)
        feat = torch.sparse.mm(x, feat)
        return feat

class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, gene_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, gene_emb=None,
                 inject_covariate=False, batch_num=None):
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()#create_activation(activation)
        self.norm0 = create_norm(norm, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            create_norm(norm, num_hidden),
        )
        if pe_type is not None:
            if cat_pe:
                num_emb = num_hidden // 2
            else:
                num_emb = num_hidden
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        # if gene_emb is None:
        #     self.feat_enc = OmicsEmbedder(gene_list, num_emb)
        # else:
        #     self.feat_enc = OmicsEmbedder(gene_list, num_emb, gene_emb)
        
        if gene_emb is None:
            self.feat_enc = OmicsEmbedderWithAttention(gene_list, num_emb)
        else:
            self.feat_enc = OmicsEmbedderWithAttention(gene_list, num_emb, gene_emb)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False

    def forward(self, x_dict, input_gene_list=None):
        # import pdb; pdb.set_trace()
        x = self.feat_enc(x_dict, input_gene_list)#self.act(self.feat_enc(x_dict, input_gene_list))
        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]
            pe = 0.#self.pe_enc(pe_input)
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict['batch'])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)
        # x = self.norm0(self.dropout(x))
        return x
