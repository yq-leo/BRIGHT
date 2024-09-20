from collections import defaultdict

import bright_utils
from new_train import train
import numpy as np
import csv
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, default='PE',
                    choices=['PE', 'F2T', 'Cora', 'Douban'],
                    help='datasets: PE; ACM-DBLP; cora; foursquare-twitter; phone-email; Douban; flickr-lastfm')
parser.add_argument('--epochs', dest='epochs', type=int, default=250, help='number of epochs')
# Experiment settings
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of runs')
parser.add_argument('--exp_name', dest='exp_name', type=str, default='exp', help='experiment name')
parser.add_argument('--edge_noise', dest='edge_noise', type=float, default=0.0, help='edge noise')
parser.add_argument('--attr_noise', dest='attr_noise', type=float, default=0.0, help='attribute noise')
parser.add_argument('--robust', dest='robust', action='store_true', help='remove metric outliers')
parser.add_argument('--record', dest='record', action='store_true', help='record results')

args = parser.parse_args()
print(args)


class Config:
    def __init__(self, data, numpy_file=None, edge_noise=0.0, attr_noise=0.0):
        self.data = data
        self.numpy_file = numpy_file

        self.dim = 128
        self.norm_g1_file = "Data/" + data + "/norm_data/network1.tsv"
        self.norm_g2_file = "Data/" + data + "/norm_data/network2.tsv"
        self.grd_truth_file = "Data/" + data + "/norm_data/grd.tsv"
        self.train_file = "Data/" + data + "/split/train_"
        self.test_file = "Data/" + data + "/split/test_"
        self.seed_file1 = "Data/" + data + "/split/seed1_"
        self.seed_file2 = "Data/" + data + "/split/seed2_"
        self.rwr1_emd = "Data/" + data + "/split/rwr_emd1_"
        self.rwr2_emd = "Data/" + data + "/split/rwr_emd2_"
        self.gcn_data = "Data/" + data + "/norm_data/gcn_data.pkl"
        self.g1_geo_data = "Data/" + data + "/norm_data/g1_geo.pkl"
        self.g2_geo_data = "Data/" + data + "/norm_data/g2_geo.pkl"

        self.edge_noise = edge_noise
        self.attr_noise = attr_noise


numpy_file_dict = {
    'Cora': 'Data/Cora/ori_data/noisy-cora1-cora2',
    'Douban': 'Data/Douban/ori_data/Douban'
}
norm = True if args.dataset in ['PE', 'F2T'] else False
use_gcn = False if args.dataset in ['PE', 'F2T'] else True

numpy_file = numpy_file_dict[args.dataset] if args.dataset in numpy_file_dict else None
config = Config(args.dataset, numpy_file, args.edge_noise, args.attr_noise)

final_hits_list = defaultdict(list)
final_mrr_list = list()
for run in range(args.runs):
    """
    get the rwr embedding 
    """
    bright_utils.preprocess(0.2, True, norm, config)
    """
    train the model
    """
    hits, mrr = train(0.2, args.epochs, 0.0001, 128, 500, 10, True, use_gcn, config)
    for key in hits:
        final_hits_list[key].append(hits[key])
    final_mrr_list.append(mrr)

final_hits, final_hits_robust = dict(), dict()
final_hits_std, final_hits_robust_std = dict(), dict()
for key in final_hits_list:
    final_hits_k = np.sort(np.array(final_hits_list[key]))
    final_hits[key] = final_hits_k.mean()
    final_hits_std[key] = final_hits_k.std()

    final_hits_k_robust = bright_utils.rm_out(final_hits_k)
    final_hits_robust[key] = final_hits_k_robust.mean()
    final_hits_robust_std[key] = final_hits_k_robust.std()

final_mrr = np.sort(np.array(final_mrr_list))
final_mrr = final_mrr.mean()
final_mrr_std = final_mrr.std()

final_mrr_list_robust = bright_utils.rm_out(final_mrr_list)
final_mrr_robust = np.sort(np.array(final_mrr_list_robust)).mean()
final_mrr_robust_std = np.sort(np.array(final_mrr_list_robust)).std()

if args.record:
    exp_name = args.exp_name
    if not os.path.exists("results"):
        os.makedirs("results")
    out_path = f"results/{exp_name}_results.csv"
    if not os.path.exists(out_path):
        with open(out_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [""] + [f"Hit@{k}" for k in final_hits] + ["MRR"] + [f"std@{k}" for k in final_hits] + [
                    "std_MRR"])

    with open(out_path, "a", newline='') as f:
        writer = csv.writer(f)
        if args.edge_noise > 0:
            header = f"{args.dataset}_({args.edge_noise:.1f})"
        else:
            header = f"{args.dataset}_({args.attr_noise:.1f})"
        writer.writerow(
            [header] + [f"{p:.3f}" for p in final_hits.values()] + [f"{final_mrr:.3f}"] + [f"{p:.3f}" for p in final_hits_std.values()] + [
                f"{final_mrr_std:.3f}"])
        if args.robust:
            writer.writerow(
                [header + "_robust"] + [f"{p:.3f}" for p in final_hits_robust.values()] + [f"{final_mrr_robust:.3f}"] + [
                    f"{p:.3f}" for p in final_hits_robust_std.values()] + [f"{final_mrr_robust_std:.10f}"])
