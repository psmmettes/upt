#
# Zero-shot action recognition using action embeddings.
#

import os
import sys
import numpy as np
import argparse

#
# Parse user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot action recognition on UCF-101")
    parser.add_argument("-v", dest="videolist", default="data/ucf-101/embeddings/ucf101_videofiles.txt", type=str)
    parser.add_argument("-a", dest="actionlist", default="data/ucf-101/embeddings/ucf101_actions.npy", type=str)
    parser.add_argument("--nr_actions", dest="nr_actions", default=101, type=int)
    parser.add_argument("--ve1", dest="video_embeddings", default="data/ucf-101/embeddings/ucf101_video_embeddings_r2plus1d18.npy", type=str)
    parser.add_argument("--ase", dest="action_source_embeddings", default="data/ucf-101/embeddings/ucf101_action_embeddings.npy", type=str)
    parser.add_argument("--ase2", dest="action_source_embeddings2", default="data/ucf-101/embeddings/ucf101_action_embeddings.npy", type=str)
    parser.add_argument("--ate", dest="action_target_embeddings", default="data/ucf-101/ot/transductive_ucf101_0_1000-kmeans_EMD.npy", type=str)
    parser.add_argument("--oe", dest="object_embeddings", default="data/ucf-101/embeddings/imagenet12988_embeddings.npy", type=str)
    parser.add_argument("--vs", dest="objectscoredir", default="data/ucf-101/objectscores/", type=str)
    parser.add_argument("-f", dest="fusion_weight", default=1, type=float)
    parser.add_argument("-t", dest="topn_objects", default=0, type=int)
    parser.add_argument("-l", dest="ilambda", default=0, type=float)
    parser.add_argument("-r", dest="resfile", default="", type=str)
    args = parser.parse_args()
    return args

#
# L2 normalization per row.
#
def l2_normalize(x):
    for i in range(x.shape[0]):
        if np.linalg.norm(x[i]) > 0:
            x[i] /= np.linalg.norm(x[i])
    return x

#
# Spherical linear interpolation.
#
def slerp(p1, p2, t):
    omega = np.arccos(np.clip(np.dot(p1, p2), -0.99999, 0.99999))
    return np.sin((t)*omega)/np.sin(omega) * p1 + np.sin((1-t) * omega) / np.sin(omega) * p2

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()
    
    # Load the UCF-101 videos and the video embeddings.
    videos = open(args.videolist).readlines()
    videos = ["/".join(v.strip().split("/")[-2:]) for v in videos]
    video_embeddings = np.load(args.video_embeddings)
    actions = [v.split("/")[0] for v in videos]
    action_embeddings = np.load(args.action_source_embeddings)
    
    # Convert action names to labels.
    action_classes = np.load(args.actionlist)
    labels = np.zeros(len(actions), dtype=int)
    subset = np.sort(np.random.choice(np.arange(len(action_classes)), args.nr_actions, replace=False))
    usedidxs = []
    for i in range(len(actions)):
        labels[i] = np.where(actions[i] == action_classes)[0]
        if labels[i] in subset:
            usedidxs.append(i)
    usedidxs = np.array(usedidxs)
    
    # Perform interpolation from source to transductive target embeddings.
    if os.path.exists(args.action_target_embeddings):
        target_embeddings = np.load(args.action_target_embeddings)
    for i in range(action_embeddings.shape[0]):
        if os.path.exists(args.action_target_embeddings):
            action_embeddings[i] = slerp(action_embeddings[i], target_embeddings[i], args.ilambda)
    
    # Pre-process the embeddings.
    video_embeddings  = l2_normalize(video_embeddings)
    action_embeddings = l2_normalize(action_embeddings)

    # Compute video to action similarity and predict most likely actions.
    v2a = np.dot(video_embeddings, action_embeddings.T)
    
    # Optional: perform object-based scoring and fusion.
    if args.topn_objects > 0 and os.path.exists(args.object_embeddings):
        object_embeddings = np.load(args.object_embeddings)
        action_embeddings2 = action_embeddings.copy()
        
        # Normalize object embeddings.
        object_embeddings = l2_normalize(object_embeddings)
        
        # Top object selection.
        top_objects = []
        a2o = np.dot(action_embeddings, object_embeddings.T)
        a2o2 = np.dot(action_embeddings2, object_embeddings.T)
        for i in range(a2o.shape[0]):
            oorder = np.argsort(a2o[i])[::-1]
            top_objects.append(oorder[:args.topn_objects])
        
        # Compute object-based action score per video.
        objectvideos = [v.split("/")[1][:-4] for v in videos]
        for i in range(len(videos)):
            if i not in usedidxs:
                continue
            # Load object scores.
            video_object_scores = np.load(args.objectscoredir + objectvideos[i] + "/avg-features.npy")
            # Compute scores.
            v2oa = np.zeros(action_embeddings.shape[0])
            for j in range(action_embeddings.shape[0]):
                v2oa[j] = np.dot(video_object_scores[top_objects[j]], (a2o2[j, top_objects[j]]))# / np.sum(a2o2[j, top_objects[j]])
            # Interpolate with action-based scores.
            v2a[i] = (args.fusion_weight) * v2a[i] + (1-args.fusion_weight) * v2oa   

    predictions = np.argmax(v2a[:,subset], axis=1)
    predictions_all = np.argsort(v2a[:,subset], axis=1)[:,::-1]
    acc_top1 = np.mean(subset[predictions[usedidxs]] == labels[usedidxs]) * 100.
    acc_top5 = np.mean([l in p for l, p in zip(labels[usedidxs], subset[predictions_all[usedidxs,:5]])]) * 100.
    s1 = args.action_target_embeddings.split("/")[-1]
    s2 = args.action_source_embeddings.split("/")[-1]
    resline = "Accuracy for %d actions (lambda-%.4f fusion-%.4f) (embeddings-%s-%s): %.4f (top 1) %.4f (top 5)" \
            %(args.nr_actions, args.ilambda, args.fusion_weight, s1, s2, acc_top1, acc_top5)
    print(resline)
