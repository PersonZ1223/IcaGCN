import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class MyLinearLayer(torch.nn.Module):
    def __init__(self, matrix, bias=False):
        super(MyLinearLayer, self).__init__()
        self.weight = torch.nn.Parameter(matrix).cuda()
        self.weight.requires_grad_(True)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(64)).cuda()
        else:
            self.register_parameter('bias', None)
    def forward(self, input):
        y = torch.sparse.mm(self.weight, input)
        return y

class SEGCNU(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SEGCNU, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']
        self.lambda_ = 1e-3
        self.gamma = 1e-4
        self.w1 = 1e-7
        self.w2 = 1
        self.w3 = 1e-7
        self.w4 = 1
        self.ii_neighbor_num = 10
        self.neg_num = 200
        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix_lgn, self.train_mat = self.get_norm_adj_mat_lgn()
        self.norm_adj_matrix_lgn = self.norm_adj_matrix_lgn.cuda()
        self.mylinearlayer = MyLinearLayer(self.norm_adj_matrix_lgn)
        self.norm_adj_matrix = self.get_norm_adj_mat().cuda()

        items_D = np.sum(self.train_mat, axis=0).reshape(-1)
        users_D = np.sum(self.train_mat, axis=1).reshape(-1)
        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        self.constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                          "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
        self.ii_neighbor_mat, self.ii_constraint_mat = self.get_ii_constraint_mat(self.train_mat, self.ii_neighbor_num)
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat_lgn(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        train_mat = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict2 = dict(zip(zip(inter_M.row, inter_M.col), [1] * inter_M.nnz))
        train_mat._update(data_dict2)
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL, train_mat

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        self.scores = self.pre_computer()
        self.scores = self.scores.cuda()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), self.scores[inter_M.col+self.n_users]))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), self.scores[inter_M_t.col])))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ii_constraint_mat(self, train_mat, num_neighbors, ii_diagonal_zero=False):
        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)  # I * I
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis=0).reshape(-1)
        users_D = np.sum(A, axis=1).reshape(-1)
        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))
        print('Computation \\Omega OK!')
        return res_mat.long(), res_sim_mat.float()

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def ms_cam(self, x):
        channels = self.n_items+self.n_users
        local_attn =x
        global_attn = x
        local_attn = self.mylinearlayer(local_attn)
        local_attn = F.relu(local_attn, inplace=True)
        local_attn = F.dropout(local_attn)
        local_attn = self.mylinearlayer(local_attn)
        local_attn = F.dropout(local_attn)
        local_attn = local_attn.cuda()
        global_attn = F.adaptive_max_pool1d(global_attn, 1)
        global_attn = self.mylinearlayer(global_attn)
        global_attn = F.relu(global_attn, inplace=True)
        global_attn = F.dropout(global_attn)
        global_attn = self.mylinearlayer(global_attn)
        global_attn = F.dropout(global_attn)
        global_attn = global_attn.cuda()
        sigmoid = nn.Sigmoid()
        gl = local_attn + global_attn
        wei = sigmoid(gl)
        wei *=2
        scores = torch.tensor(wei)
        scores = F.adaptive_max_pool1d(scores, 1)
        scores = scores.reshape(channels, 1)
        return scores

    def pre_computer(self):
        all_embeddings = self.get_ego_embeddings()
        all_embeddings = all_embeddings.cuda()
        scores = self.ms_cam(all_embeddings)
        return scores

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        all_embeddings = all_embeddings.cuda()
        self.norm_adj_matrix = self.norm_adj_matrix.cuda()
        #for layer_idx in range(self.n_layers):
        #    all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
        #    embeddings_list.append(all_embeddings)
        all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
        embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def get_omegas(self, users, pos_items, neg_items):

        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).cuda()
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).cuda()
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                                   self.constraint_mat['beta_iD'][neg_items.flatten()]).cuda()
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).cuda()
        weight = torch.cat((pow_weight, neg_weight))

        return weight

    def cal_loss_L(self, user_embeds, pos_embeds, neg_embeds, omega_weight):

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num
        neg_labels = torch.zeros(neg_scores.size()).cuda()
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction='none').mean(dim=-1)
        #, weight=omega_weight[len(pos_scores):].view(neg_scores.size())
        pos_labels = torch.ones(pos_scores.size()).cuda()
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)], reduction='none')
        #, weight=omega_weight[:len(pos_scores)]
        loss = pos_loss + neg_loss

        return loss.sum()

    def cal_loss_I(self, user_embeds, sim_scores, neighbor_embeds):
        #t3 = time.time()
        user_embeds = user_embeds.unsqueeze(1)
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        #print(f'cal_loss_I:{time.time() - t3:.4f}s')
        return loss.sum()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]

        # calculate UltraGCN Loss
        neg_items = []
        neg_candidates = np.arange(self.n_items)
        for i in pos_item:
            probs = np.ones(self.n_items)
            probs[i] = 0
            probs /= np.sum(probs)
            u_neg_items = np.random.choice(neg_candidates, size=self.neg_num, p=probs, replace=True).reshape(1, -1)
            neg_items.append(u_neg_items)
        neg_items = np.concatenate(neg_items, axis=0)
        neg_items = torch.from_numpy(neg_items).cuda()
        #neg_embeddings = item_all_embeddings[neg_items]
        #t1 = time.time()
        #omega_weight = self.get_omegas(user, pos_item, neg_items)
        #print(f'get_omegas:{time.time() - t1:.4f}s')
        #t2 = time.time()
        #mf_loss = self.cal_loss_L(u_embeddings, pos_embeddings, neg_embeddings, omega_weight)
        #print(f'cal_loss_L:{time.time() - t2:.4f}s')
        neighbor_embeds = item_all_embeddings[self.ii_neighbor_mat[pos_item].cuda()]
        sim_scores = self.ii_constraint_mat[pos_item].cuda()
        loss_i = self.cal_loss_I(u_embeddings, sim_scores, neighbor_embeds)
        #mf_loss += self.gamma * mf_loss
        mf_loss = self.lambda_ * loss_i
        return mf_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
