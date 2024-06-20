import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best_recall, best_precision, best_ndcg = 0.0, 0.0, 0.0
    recall_epoch, precision_epoch, ndcg_epoch = 0, 0, 0

    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            _, recall, precision, ndcg = Procedure.Test(
                dataset, Recmodel, epoch, w, world.config["multicore"]
            )
            if recall > best_recall:
                best_recall = recall
                recall_epoch = epoch
            if precision > best_precision:
                best_precision = precision
                precision_epoch = epoch
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                ndcg_epoch = epoch

        output_information = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
        )
        print(
            f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}, best recall e{recall_epoch}:{best_recall:.5f}, best precision e{precision_epoch}:{best_precision:.5f}, best ndcg e{ndcg_epoch}:{best_ndcg:.5f}"
        )
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
