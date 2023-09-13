from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import device


class VGAEModel(nn.Module):
    def __init__(self, in_dim, drug_num, num_relations, hidden1_dim, hidden2_dim, reg_param=0.01):
        super(VGAEModel, self).__init__()
        self.in_dim = in_dim
        self.reg_param = reg_param
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.sampled_z = nn.Embedding(drug_num, hidden2_dim)
        # 用于DistMult相乘的矩阵R，随机初始化,这里的维度是所有关系的维度，hidden2_dim是隐变量输出的维度
        self.w_relation = nn.Parameter(torch.Tensor(num_relations, self.hidden2_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

        layers = [GraphConv(self.in_dim, self.hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                  GraphConv(self.hidden1_dim, self.hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True)]
        self.layers = nn.ModuleList(layers)

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(device)
        return sampled_z

    def get_score(self, s, r, o):
        return torch.sum(s * r * o, dim=1)

    def distmult(self, embedding, triplets):
        # 这里的embedding是所有节点的，所以没有类似于RGCN取出相应triplets中的节点特征
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 2]]
        o = embedding[triplets[:, 1]]
        score = self.get_score(s, r, o)
        return score

    def reg_loss(self, embedding):

        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def decoder(self, z, drugs_num, triplets):
        # drug节点的embedding
        z_drug = z[:drugs_num]
        # target节点的embedding
        z_dis = self.w_relation.data
        # triplets decoder
        tri_score = self.distmult(z_drug, triplets)
        # drug-target decoder
        d_j_recd = torch.sigmoid(torch.matmul(z_drug, z_dis.t()))
        # target_target decoder
        j_j_recd = torch.sigmoid(torch.matmul(z_dis, z_dis.t()))
        # reg_loss
        reg_loss = self.reg_param * self.reg_loss(z_drug)

        return tri_score, d_j_recd, j_j_recd, reg_loss
        # return tri_score, d_j_recd, reg_loss
        # return tri_score, j_j_recd, reg_loss

    def forward(self, g, features, drug_num, train_triplets):
        z = self.encoder(g, features)
        self.sampled_z = nn.Parameter(z[:drug_num])
        tri_score, d_j_recd, j_j_recd, reg_loss = self.decoder(z, drug_num, train_triplets)  # 解码器还原邻接矩阵
        # tri_score, d_j_recd, reg_loss = self.decoder2(z, drug_num, train_triplets)  # 解码器还原邻接矩阵
        # tri_score, j_j_recd, reg_loss = self.decoder2(z, drug_num, train_triplets)  # 解码器还原邻接矩阵
        return tri_score, d_j_recd, j_j_recd, reg_loss, z
        # return tri_score, d_j_recd, reg_loss, z
        # return tri_score, j_j_recd, reg_loss, z


