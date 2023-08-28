#
# Compute optimal transport destinations for each element in a source embedding.
#

import os
import sys
import numpy as np
import argparse
import ot
import ot.plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.hypersphere import Hypersphere
import time, datetime


#
# L2 normalization per row.
#
def l2_normalize(x):
    for i in range(x.shape[0]):
        if np.linalg.norm(x[i]) > 0:
            x[i] /= np.linalg.norm(x[i])
    return x

#
# Parse user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Optimal transport of embeddings")
    parser.add_argument("--source", dest="semb", default="data/ucf-101/embeddings/ucf101_action_embeddings.npy", type=str)
    parser.add_argument("--target", dest="temb", default="data/ucf-101/embeddings/ucf101_video_embeddings_r2plus1d18.npy", type=str)
    parser.add_argument("--mapper", dest="mapper", default="EMD", type=str)
    parser.add_argument("--cluster", dest="cluster", default="kmeans", type=str)
    parser.add_argument("-n", dest="nr_clusters", default=0, type=int)
    parser.add_argument("-t", dest="topn", default=0, type=int)
    parser.add_argument("--resname", dest="resname", default="", type=str)
    args = parser.parse_args()
    return args

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()
    
    # Load embeddings.
    semb = l2_normalize(np.load(args.semb))
    temb = l2_normalize(np.load(args.temb))
    #stot = np.dot(semb, temb.T)
    print(semb.shape, temb.shape)
    
    # Perform spherical k-means on the target embeddings.
    if args.cluster == "none" or args.nr_clusters == temb.shape[0]:
        print("Not clustering")
        temb_clustered = temb.copy()
    elif args.cluster == "kmeans" or args.cluster == "wkmeans":
        clustering = KMeans(n_clusters=args.nr_clusters, n_init=10).fit(temb)
        temb_clustered = l2_normalize(clustering.cluster_centers_)
    else:
        print("Invalid cluster method, exiting.")
        exit()
    print(temb_clustered.shape)
    
    # Optionally: remove target embeddings outside of any topn.
    if args.topn > 0:
        print("Doing topn selection!")
        tokeep = np.array([], dtype=int)
        e2c  = np.dot(semb, temb_clustered.T)
        for i in range(e2c.shape[0]):
            eorder = np.argsort(e2c[i,:])[::-1]
            tokeep = np.concatenate((tokeep, eorder[:args.topn]))
        tokeep = np.unique(tokeep)
        temb_clustered = temb_clustered[tokeep,:]
        print(temb_clustered.shape, e2c.shape)
    else:
        tokeep = np.arange(temb_clustered.shape[0])
    
    # Compute and normalize the costs.
    e2c  = 1 - np.dot(semb, temb_clustered.T)
    e2c /= np.max(e2c)
    
    # Compute mapping.
    w1 = np.ones(semb.shape[0]) / float(semb.shape[0])
    if args.cluster[0] == "w":
        w2  = np.bincount(clustering.labels_, minlength=args.nr_clusters).astype(float)
        w2  = w2[tokeep]
        w2 /= np.sum(w2)
    else:
        w2 = np.ones(temb_clustered.shape[0]) / float(temb_clustered.shape[0])
    if args.mapper == "EMD":
        mapping = ot.emd(w1, w2, e2c)
    elif args.mapper == "sinkhorn":
        mapping = ot.sinkhorn(w1, w2, e2c, 1e-2)
    else:
        print("Invalid OT method, exiting.")
        exit()

    # Frechet mean initialization.
    sphere = Hypersphere(dim=semb.shape[1] - 1)
    estimator = FrechetMean(metric=sphere.metric)
    
    #time2 = time.time()
    time2 = datetime.datetime.now()

    # Determine target embeddings.
    target = np.zeros(semb.shape)
    for i in range(target.shape[0]):
        weights  = mapping[i]
        weights /= np.sum(weights)
        target[i] = estimator.fit(temb_clustered, weights=weights).estimate_
        target[i] /= np.linalg.norm(target[i])
    print(target.shape)
    
    # Store results.
    np.save(args.resname + "%d_%d-%s_%s.npy" %(args.topn, args.nr_clusters, args.cluster, args.mapper), target)
    print("Results stored at: " + args.resname + "%d_%d-%s_%s.npy" %(args.topn, args.nr_clusters, args.cluster, args.mapper))
