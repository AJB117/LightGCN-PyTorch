import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ["gowalla", "yelp2018", "amazon-book", "Movielens1M", "lastfm"]:
    if world.config["model"] == "lgnliteuser":
        dataset = dataloader.LoaderUserUser(path="../data/" + world.dataset)
    elif world.config["model"] == "lgndecoupled":
        dataset = dataloader.LoaderDecoupled(path="../data/" + world.dataset)
    else:
        dataset = dataloader.Loader(path="../data/" + world.dataset)
elif world.dataset == "lastfm":
    dataset = dataloader.LastFM()

print("===========config================")
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print("===========end===================")

MODELS = {
    "mf": model.PureMF,
    "lgn": model.LightGCN,
    "lgnlite": model.LightGCNLite,
    "lgnliteuser": model.LightGCNLiteUser,
    "lgndecoupled": model.LightGCNDecoupled,
}
