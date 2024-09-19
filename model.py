import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv
import numpy as np
import scipy
import time


class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hid_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class BRIGHT_A(torch.nn.Module):

    def __init__(self, dim, rwr_dim, g1_feat_dim):
        super(BRIGHT_A, self).__init__()
        self.lin = torch.nn.Linear(rwr_dim, dim)
        self.gcn1 = GCN(g1_feat_dim, dim)
        self.combine1 = torch.nn.Linear(2*dim, dim)


    def forward(self, rwr1_emd, rwr2_emd, data1, data2):
            pos_emd1 = self.lin(rwr1_emd)
            pos_emd2 = self.lin(rwr2_emd)
            gcn_emd1 = self.gcn1(data1)
            gcn_emd2 = self.gcn1(data2)
            pos_emd1 = F.normalize(pos_emd1, p=1, dim=1)
            pos_emd2 = F.normalize(pos_emd2, p=1, dim=1)
            gcn_emd1 = F.normalize(gcn_emd1, p=1, dim=1)
            gcn_emd2 = F.normalize(gcn_emd2, p=1, dim=1)
            emd1 = torch.cat([pos_emd1, gcn_emd1], 1)
            emd2 = torch.cat([pos_emd2, gcn_emd2], 1)
            emd1 = self.combine1(emd1)
            emd1 = F.normalize(emd1, p=1, dim=1)
            emd2 = self.combine1(emd2)
            emd2 = F.normalize(emd2, p=1, dim=1)
            return emd1, emd2


class BRIGHT_U(torch.nn.Module):

    def __init__(self, dim, rwr_dim):
        super(BRIGHT_U, self).__init__()
        self.lin1 = torch.nn.Linear(rwr_dim, dim)

    def forward(self, rwr1_emd, rwr2_emd):
        pos_emd1 = self.lin1(rwr1_emd)
        pos_emd2 = self.lin1(rwr2_emd)
        pos_emd1 = F.normalize(pos_emd1, p=1, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=1, dim=1)
        return pos_emd1, pos_emd2

"""
BRIGHT with just GCN
"""
class BRIGHT_gcn(torch.nn.Module):

    def __init__(self, dim, g1_feat_dim, g2_feat_dim):
        super(BRIGHT_gcn, self).__init__()
        self.gcn1 = GCN(g1_feat_dim, dim)
        self.gcn2 = GCN(g2_feat_dim, dim)

    def forward(self, data1, data2):
        gcn_emd1 = self.gcn1(data1)
        gcn_emd2 = self.gcn1(data2)
        gcn_emd1 = F.normalize(gcn_emd1, p=1, dim=1)
        gcn_emd2 = F.normalize(gcn_emd2, p=1, dim=1)
        return gcn_emd1, gcn_emd2


class ranking_loss_L1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out1, out2, anchor1, anchor2, neg1, neg2, gamma):
        anchor1_vec = out1[anchor1]
        anchor2_vec = out2[anchor2]
        neg1_vec = out2[neg1]
        neg2_vec = out1[neg2]

        A = torch.sum(torch.abs(anchor1_vec - anchor2_vec), 1)
        D = A + gamma
        B1 = -torch.sum(torch.abs(anchor1_vec - neg1_vec), 1)
        L1 = torch.sum(F.relu(B1 + D))
        B2 = -torch.sum(torch.abs(anchor2_vec - neg2_vec), 1)
        L2 = torch.sum(F.relu(B2 + D))
        return (L1 + L2)/len(anchor1)


class RankingLossL1New(torch.nn.Module):
    def __init__(self, k, gamma):
        super().__init__()
        self.k = k
        self.gamma = gamma

    def neg_sampling(self, out1, out2, anchor1, anchor2):
        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]

        distances_1 = scipy.spatial.distance.cdist(anchor_embeddings_1, out2, metric='cityblock')
        ranks_1 = np.argsort(distances_1, axis=1)
        neg_samples_1 = ranks_1[:, :self.k]

        distances_2 = scipy.spatial.distance.cdist(anchor_embeddings_2, out1, metric='cityblock')
        ranks_2 = np.argsort(distances_2, axis=1)
        neg_samples_2 = ranks_2[:, :self.k]

        return neg_samples_1, neg_samples_2

    def forward(self, out1, out2, anchor1, anchor2):
        start = time.time()

        np_out1 = out1.detach().cpu().numpy()
        np_out2 = out2.detach().cpu().numpy()
        anchor1 = np.array(anchor1)
        anchor2 = np.array(anchor2)

        neg1, neg2 = self.neg_sampling(np_out1, np_out2, anchor1, anchor2)

        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]
        neg_embeddings_1 = out2[neg1, :]
        neg_embeddings_2 = out1[neg2, :]

        A = torch.sum(torch.abs(anchor_embeddings_1 - anchor_embeddings_2), 1)
        D = A + self.gamma
        B1 = -torch.sum(torch.abs(anchor_embeddings_1.unsqueeze(1) - neg_embeddings_1), 2)
        L1 = torch.sum(F.relu(D.unsqueeze(-1) + B1))
        B2 = -torch.sum(torch.abs(anchor_embeddings_2.unsqueeze(1) - neg_embeddings_2), 2)
        L2 = torch.sum(F.relu(D.unsqueeze(-1) + B2))

        print("Time taken: ", time.time() - start)

        return (L1 + L2) / (anchor1.shape[0] * self.k)


class RankingLossL1(torch.nn.Module):
    def __init__(self, k, gamma):
        super().__init__()
        self.k = k
        self.gamma = gamma

    def get_neg(self, out1, out2, anchor1, anchor2):
        neg1 = []
        neg2 = []
        t = len(anchor1)
        anchor1_vec = np.array(out1[anchor1])
        anchor2_vec = np.array(out2[anchor2])
        G1_vec = np.array(out1)
        G2_vec = np.array(out2)
        sim1 = scipy.spatial.distance.cdist(anchor1_vec, G2_vec, metric='cityblock')
        for i in range(t):
            rank = sim1[i, :].argsort()
            neg1.append(rank[0: self.k])
        neg1 = np.array(neg1)
        neg1 = neg1.reshape((t * self.k,))
        sim2 = scipy.spatial.distance.cdist(anchor2_vec, G1_vec, metric='cityblock')
        for i in range(t):
            rank = sim2[i, :].argsort()
            neg2.append(rank[0:self.k])
        anchor1 = np.repeat(anchor1, self.k)
        anchor2 = np.repeat(anchor2, self.k)
        neg2 = np.array(neg2)
        neg2 = neg2.reshape((t * self.k,))
        return anchor1, anchor2, neg1, neg2

    def forward(self, out1, out2, anchor1_org, anchor2_org):
        start = time.time()
        np_out1 = out1.detach().cpu().numpy()
        np_out2 = out2.detach().cpu().numpy()
        anchor1, anchor2, neg1, neg2 = self.get_neg(np_out1, np_out2, anchor1_org, anchor2_org)
        # anchor1_m, anchor2_m, neg1_m, neg2_m = self.neg_sampling(np_out1, np_out2, anchor1_org.copy(), anchor2_org.copy())
        # assert np.array_equal(anchor1, anchor1_m), "Anchor1 not equal"
        # assert np.array_equal(anchor2, anchor2_m), "Anchor2 not equal"
        # assert np.array_equal(neg1, neg1_m), "Neg1 not equal"
        # assert np.array_equal(neg2, neg2_m), "Neg2 not equal"

        # loss_val = self.loss(out1, out2, anchor1_m, anchor2_m, neg1_m, neg2_m)

        anchor1_vec = out1[anchor1]
        anchor2_vec = out2[anchor2]
        neg1_vec = out2[neg1]
        neg2_vec = out1[neg2]

        A = torch.sum(torch.abs(anchor1_vec - anchor2_vec), 1)
        D = A + self.gamma
        B1 = -torch.sum(torch.abs(anchor1_vec-neg1_vec), 1)
        L1 = torch.sum(F.relu(B1 + D))
        B2 = -torch.sum(torch.abs(anchor2_vec - neg2_vec), 1)
        L2 = torch.sum(F.relu(B2 + D))

        print("Time taken: ", time.time() - start)

        # print("Loss: ", loss_val, loss_org)
        # print("Min Element1: ", torch.min(out1))
        # print("Min Element2: ", torch.min(out2))
        # assert loss_val.item() == loss_org.item(), "Loss not equal"

        return (L1 + L2)/len(anchor1)
