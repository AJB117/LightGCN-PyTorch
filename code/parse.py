"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument(
        "--bpr_batch",
        type=int,
        default=2048,
        help="the batch size for bpr loss training procedure",
    )
    parser.add_argument(
        "--recdim", type=int, default=64, help="the embedding size of lightGCN"
    )
    parser.add_argument(
        "--layer", type=int, default=3, help="the layer num of lightGCN"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="the learning rate")
    parser.add_argument(
        "--decay", type=float, default=1e-4, help="the weight decay for l2 normalizaton"
    )
    parser.add_argument(
        "--dropout", type=int, default=0, help="using the dropout or not"
    )
    parser.add_argument(
        "--keepprob",
        type=float,
        default=0.6,
        help="the batch size for bpr loss training procedure",
    )
    parser.add_argument(
        "--a_fold",
        type=int,
        default=100,
        help="the fold num used to split large adj matrix, like gowalla",
    )
    parser.add_argument(
        "--testbatch", type=int, default=100, help="the batch size of users for testing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gowalla",
        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]",
    )
    parser.add_argument(
        "--path", type=str, default="./checkpoints", help="path to save weights"
    )
    parser.add_argument("--topks", nargs="?", default="[20]", help="@k test list")
    parser.add_argument("--tensorboard", type=int, default=1, help="enable tensorboard")
    parser.add_argument("--comment", type=str, default="lgn")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--multicore",
        type=int,
        default=0,
        help="whether we use multiprocessing or not in test",
    )
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="whether we use pretrained weight or not",
    )
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="lgn",
        choices=["mf", "lgn", "lgnlite", "lgnliteuser", "lgndecoupled"],
        help="rec-model, support [mf, lgn, lgnlite, lgnliteuser, lgndecoupled]",
    )
    parser.add_argument("--device", type=int, default=0, help="gpu id")

    # MINE
    parser.add_argument(
        "--lite", action="store_true", help="whether to omit the item embeddings"
    )
    parser.add_argument(
        "--item_transform_linear",
        action="store_true",
        help="whether to use a linear transformation for items",
    )
    parser.add_argument(
        "--item_transform_mlp",
        action="store_true",
        help="whether to use a mlp transformation for items",
    )
    parser.add_argument("--spectral_dim", default=64)
    parser.add_argument("--spectral_reweight", nargs="+", type=float, default=[1.0])
    parser.add_argument("--zero_diag", action="store_true")
    parser.add_argument("--sort_direction", type=str, default="descending")
    parser.add_argument("--solver", type=str, default="svd")
    parser.add_argument(
        "--use_which",
        type=str,
        default="both",
        help="whether to use both users and item embedding propagation or just one",
        choices=["both", "users", "items"],
    )
    parser.add_argument(
        "--use_grad_which",
        type=str,
        default="both",
        help="whether to learn both user and item embeddings or just one",
        choices=["both", "users", "items"],
    )
    parser.add_argument("--no_init_residual", help="whether to add the initial heterogeneous round of MP embeddings", action="store_false")
    parser.add_argument("--initialization", type=str, default="normal", help="initialization method")

    return parser.parse_args()
