"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""

import scipy.sparse as sp
import pdb
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (
            (1 / 2)
            * (
                users_emb.norm(2).pow(2)
                + pos_emb.norm(2).pow(2)
                + neg_emb.norm(2).pow(2)
            )
            / float(len(users))
        )
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        # self.orth_user = torch.rand(
        #     self.num_users, self.latent_dim, requires_grad=False
        # ).to(world.device)
        # self.orth_item = torch.rand(
        #     self.num_items, self.latent_dim, requires_grad=False
        # ).to(world.device)
        # self.transform = nn.Linear(self.latent_dim // 16, self.latent_dim)

        if self.config["pretrain"] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"])
            )
            # self.embedding_item.weight.data.copy_(
            #     torch.from_numpy(self.config["item_emb"])
            # )
            print("use pretrained data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        # users_emb = torch.cat([self.embedding_user.weight, self.orth_user], dim=-1)
        # items_emb = torch.cat([self.embedding_item.weight, self.orth_item], dim=-1)
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config["dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCNLite(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCNLite, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]

        if self.config["item_transform_linear"]:
            self.item_transform = nn.Linear(self.latent_dim, self.latent_dim)
        elif self.config["item_transform_mlp"]:
            self.item_transform = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.zeros(
            self.num_items, self.latent_dim, requires_grad=False
        ).to(world.device)
        # self.embedding_user = torch.zeros(self.num_users, self.latent_dim).to(
        #     world.device
        # )
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.latent_dim
        # )

        if self.config["pretrain"] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            world.cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"])
            )
            # self.embedding_item.weight.data.copy_(
            #     torch.from_numpy(self.config["item_emb"])
            # )
            print("use pretrained data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config["dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for idx, layer in enumerate(range(self.n_layers)):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
                # users_emb, item_emb = torch.split(
                #     all_emb, [self.num_users, self.num_items]
                # )
                # item_emb = self.item_transform(item_emb)
                # all_emb = torch.cat([users_emb, item_emb])

            embs.append(all_emb)

        # embs = torch.stack(embs, dim=1)
        # # print(embs.size())
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])

        # pdb.set_trace()
        new_embs = []
        for emb in embs:
            _, item_emb = torch.split(emb, [self.num_users, self.num_items])
            item_emb = self.item_transform(item_emb)
            emb = torch.cat([users_emb, item_emb])
            new_embs.append(emb)

        embs = new_embs

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # items = self.item_transform(items)

        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item[pos_items]
        neg_emb_ego = self.embedding_item[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCNLiteUser(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCNLiteUser, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]

        if self.config["item_transform_linear"]:
            self.item_transform = nn.Linear(self.latent_dim, self.latent_dim)
        elif self.config["item_transform_mlp"]:
            self.item_transform = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        if self.config["pretrain"] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            world.cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"])
            )
            # self.embedding_item.weight.data.copy_(
            #     torch.from_numpy(self.config["item_emb"])
            # )
            print("use pretrained data")
        self.f = nn.Sigmoid()
        self.UserUserGraph = self.dataset.getSparseUserUserGraph()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        self.item_zeros = torch.zeros(self.num_items, self.latent_dim).to(world.device)

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.UserUserGraph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.UserUserGraph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight

        all_emb = users_emb
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config["dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.UserUserGraph
        else:
            g_droped = self.UserUserGraph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)

            embs.append(all_emb)

        # embs = torch.stack(embs, dim=1)
        # # print(embs.size())
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])

        # pdb.set_trace()
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users = light_out
        # items = self.item_transform(items)

        return users

    def getItemEmbeddings(self, users_embs):
        items_zeros = self.item_zeros

        all_emb = torch.cat([users_embs, items_zeros])
        all_emb = torch.sparse.mm(self.Graph, all_emb)
        _, items = torch.split(all_emb, [self.num_users, self.num_items])

        items_emb = self.item_transform(items)

        return items

    def getUsersRating(self, users):  # only used in test
        all_users = self.computer()
        users_emb = all_users[users.long()]
        items = torch.arange(self.num_items).to(world.device)
        items_emb = self.getItemEmbeddings(all_users)[items]
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users = self.computer()
        users_emb = all_users[users]

        # pos_users = self.Graph[pos_items].sum(dim=1)
        # neg_users = self.Graph[neg_items].sum(dim=1)
        item_embs = self.getItemEmbeddings(all_users)
        pos_users = item_embs[pos_items]
        neg_users = item_embs[neg_items]

        # pos_emb = all_items[pos_items]
        # neg_emb = all_items[neg_items]
        pos_emb = pos_users
        neg_emb = neg_users

        users_emb_ego = self.embedding_user(users)
        # pos_emb_ego = self.embedding_item(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        pos_emb_ego = torch.tensor(0.0)
        neg_emb_ego = torch.tensor(0.0)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        # neigh_users = self.Graph[items].sum(dim=1)
        items_emb = self.getItemEmbeddings(all_users, items)

        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class LightGCNDecoupled(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCNDecoupled, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.spectral_dim = int(config["spectral_dim"])
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]

        if self.config["item_transform_linear"]:
            self.item_transform = nn.Linear(self.latent_dim, self.latent_dim)
        elif self.config["item_transform_mlp"]:
            self.item_transform = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        ).to(world.device)
        # self.embedding_item = torch.zeros(
        #    self.num_items, self.latent_dim, requires_grad=False
        # ).to(world.device)
        # self.embedding_user = torch.zeros(self.num_users, self.latent_dim).to(
        #     world.device
        # )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        ).to(world.device)

        if self.config["pretrain"] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"])
            )
            # self.embedding_item.weight.data.copy_(
            #     torch.from_numpy(self.config["item_emb"])
            # )
            print("use pretrained data")
        self.f = nn.Sigmoid()
        self.Graph, self.ItemGraph, self.UserGraph = self.dataset.getSparseGraph()
        UserEigs, ItemEigs = self.dataset.getHomogeneousEigs(
            k=self.config["spectral_dim"]
        )
        self.UserEigvals, self.UserEigvecs = UserEigs
        self.ItemEigvals, self.ItemEigvecs = ItemEigs

        # # convert usergraph to scipy sparse matrix
        # self.UserGraph = sp.coo_matrix(self.UserGraph.cpu().to_dense().numpy())

        # # self.graph_prods = [self.UserGraph.to(torch.device("cpu"))]
        # self.graph_prods = [self.UserGraph]
        # for i in range(self.n_layers):
        #     # print(sp.coo_matrix.multiply(self.graph_prods[-1], self.UserGraph.T))
        #     if i % 2 == 0:
        #         self.graph_prods.append(
        #             (self.graph_prods[-1] @ self.UserGraph.T).tocoo().astype(np.float32)
        #         )
        #     else:
        #         self.graph_prods.append(
        #             (self.graph_prods[-1] @ self.UserGraph).tocoo().astype(np.float32)
        #         )

        # # convert all the graph_prods to torch.sparse.FloatTensor
        # for i in range(len(self.graph_prods)):
        #     self.graph_prods[i] = torch.sparse.FloatTensor(
        #         torch.tensor(
        #             [self.graph_prods[i].row, self.graph_prods[i].col], dtype=torch.long
        #         ),
        #         torch.tensor(self.graph_prods[i].data, dtype=torch.float),
        #         torch.Size(self.graph_prods[i].shape),
        #     ).to(world.device)

        self.spectral_reweight = self.config["spectral_reweight"]
        print(f"reweighting spectral radius by {self.spectral_reweight}")

        # I = torch.eye(self.spectral_dim).to(world.device)
        # low_weighting = 0.34 * (1 - self.UserEigvals)
        # med_weighting = 0.33 * (self.UserEigvals - 1) ** 2
        # high_weighting = 0.33 * (self.UserEigvals + 1)

        # self.UserEigvals = (
        #     low_weighting + med_weighting + high_weighting
        # )
        # self.ItemEigvals = self.UserEigvals

        self.UserEigvals_geom = self.spectral_reweight / (
            1 - self.UserEigvals * self.spectral_reweight
        )  # geometric series
        self.ItemEigvals_geom = self.spectral_reweight / (
            1 - self.ItemEigvals * self.spectral_reweight
        )

        self.user_eigval_weights = nn.Parameter(torch.ones(self.spectral_dim)).to(
            world.device
        )
        self.item_eigval_weights = nn.Parameter(
            torch.ones(self.spectral_dim), requires_grad=True
        ).to(world.device)

        nn.init.uniform_(self.user_eigval_weights, a=0, b=1)
        nn.init.uniform_(self.item_eigval_weights, a=0, b=1)

        # self.user_ortho_matrix = nn.Parameter(
        #     torch.eye(self.num_users, self.spectral_dim), requires_grad=True
        # ).to(world.device)
        self.user_ortho_matrix = torch.rand(
            self.num_users, self.spectral_dim, requires_grad=False
        ).to(world.device)
        nn.init.orthogonal_(self.user_ortho_matrix)

        # self.item_ortho_matrix = nn.Parameter(
        #     torch.eye(self.num_items, self.spectral_dim), requires_grad=True
        # ).to(world.device)
        self.item_ortho_matrix = torch.rand(
            self.num_items, self.spectral_dim, requires_grad=False
        ).to(world.device)
        nn.init.orthogonal_(self.item_ortho_matrix)

        self.UserUserGraph = self.dataset.ItemItemGraph  # messed up naming convention
        self.ItemItemGraph = self.dataset.UserUserGraph

        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]

        if self.config["dropout"]:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        initial_user_emb = (
            self.embedding_user.weight + self.ItemGraph @ self.embedding_item.weight
        )
        initial_item_emb = (
            self.embedding_item.weight + self.UserGraph @ self.embedding_user.weight
        )

        user_eigvals = self.UserEigvals_geom
        item_eigvals = self.ItemEigvals_geom

        user_emb = initial_user_emb + self.UserEigvecs @ (
            user_eigvals.unsqueeze(1) * (self.UserEigvecs.t() @ initial_user_emb)
        )
        # user_eigval_weights = self.spectral_reweight / (
        #     1 - self.user_eigval_weights * self.spectral_reweight
        # ).view(1, -1)
        # item_eigval_weights = self.spectral_reweight / (
        #     1 - self.item_eigval_weights * self.spectral_reweight
        # ).view(1, -1)
        # new_user_weights = self.user_eigval_weight_mlp(user_eigval_weights)
        # new_item_weights = self.item_eigval_weight_mlp(item_eigval_weights)

        # user_emb = initial_user_emb + self.user_ortho_matrix @ (
        #     new_user_weights.unsqueeze(1) * (self.user_ortho_matrix.t() @ initial_user_emb)
        # )
        # user_emb = initial_user_emb
        # user_emb = initial_user_emb + self.UserProd @ initial_user_emb
        item_emb = initial_item_emb + self.ItemEigvecs @ (
            item_eigvals.unsqueeze(1) * (self.ItemEigvecs.t() @ initial_item_emb)
        )
        # item_emb = initial_item_emb
        # item_emb = initial_item_emb + self.ItemProd @ initial_item_emb

        # embs = torch.stack(embs, dim=1)
        # # print(embs.size())
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])

        # pdb.set_trace()

        # embs = torch.stack(embs, dim=1)
        # light_out = torch.mean(embs, dim=1)
        # users, items = torch.split(light_out, [self.num_users, self.num_items])

        users, items = user_emb, item_emb

        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
                # + self.user_eigval_weights.norm(2).pow(2)
                # + self.item_eigval_weights.norm(2).pow(2)
            )
            / float(len(users))
        )

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
