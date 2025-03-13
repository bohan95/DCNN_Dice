#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19, 2024

Modified by Soumyanil Banerjee on June 20, 2024
Tract Classification Project
- Added KL-divergence for Self-Distillation and clustering loss
- Added klDic.py and clustering_layer.py and modified main_40_16_GPU_KL.py file
- Updated RESNET152_ATT_naive.py file to include bottleneck layers for Self-Distillation (both at class level and feature-map level)

@author: Soumyanil Banerjee
"""
import RESNET152_ATT_naive
from CenterLoss import CenterLoss
from klDiv import KLDivLoss
from clustering_layer_v2 import ClusterlingLayer
from ADAMW import AdamW

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pickle
import gc
import re
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

GPUINX='0'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=GPUINX
np.random.seed(987)
device = torch.device("cuda:{}".format(GPUINX) if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
"""v flip and shuffle"""
# def udflip(X_nparray,y_nparray,shuffle=True):
#     output=np.zeros((X_nparray.shape[0],X_nparray.shape[1],X_nparray.shape[2]),dtype=np.float32)
#     for i in range(X_nparray.shape[0]):
#         # a = X_nparray[i,:] # just for checking the code
#         output[i,:]=np.flipud(X_nparray[i,:])
#     output=np.vstack((X_nparray,output))
#     y=np.hstack((y_nparray,y_nparray))
#     if shuffle:
#         shuffle_inx=np.random.permutation(output.shape[0])
#         return output[shuffle_inx],y[shuffle_inx]
#     else:
#         return output,y
def udflip(X_nparray, y_nparray, shuffle=True):

    if X_nparray.shape[2] == 4:
        if np.std(X_nparray[:, 0, :]) > np.std(X_nparray[:, -1, :]):
            print("Detected special info in first column, swapping...")
            X_nparray = np.concatenate((X_nparray[:, 1:, :], X_nparray[:, 0:1, :]), axis=1)
    
    X_flipped = np.flip(X_nparray, axis=2)  

    X_aug = np.vstack((X_nparray, X_flipped))
    y_aug = np.hstack((y_nparray, y_nparray))  

    if shuffle:
        shuffle_idx = np.random.permutation(X_aug.shape[0])
        return X_aug[shuffle_idx], y_aug[shuffle_idx]
    else:
        return X_aug, y_aug


def aug_at_test(probs,mode='max'):
    assert(len(probs)>0)
    if(mode=='max'):
        all_probs=np.vstack(probs)
        max_probs=np.amax(all_probs,axis=1).reshape((2,-1))#row 0: prob for first half, row 1: prob for flipped half
        max_idx=np.argmax(max_probs,axis=0)#should be 0/1
        test_sample_count=all_probs.shape[0]/2
        
        class_pred=np.argmax(all_probs,axis=1)
        final_pred=list()
        for i in range(max_idx.shape[0]):
            final_pred.append(class_pred[int(i+test_sample_count*max_idx[i])])#if 0, first half
        return final_pred
    if(mode=='mean'):
        all_probs=np.exp(np.vstack(probs))
        test_sample_count=int(all_probs.shape[0]/2)
        final_probs=all_probs[0:test_sample_count]+all_probs[test_sample_count:]
        final_pred=np.argmax(final_probs,axis=1)
        return final_pred.tolist()

def datato3d(arrays):#list of np arrays, no_tractsx4x100
    output=list()
    for i in arrays:
        i=np.transpose(i,(0,2,1))
        output.append(i)
    return output

def compute_cluster_anatomical_profile(preds, fiber_rois, num_clusters, num_anatomical_rois, threshold=0.4):
    """
    compute cluster's anatomical profile。

    - preds: (batch_size,) -> fiber predect cluster id
    - fiber_rois: (batch_size, num_fiber_points) -> each fiber passed anatomical profile
    - num_clusters: cluster number
    - num_anatomical_rois: ROI number
    - threshold: filter ROI threshold（default 40%）

    return: (num_clusters, num_anatomical_rois) -> cluster_anatomical_profile
    """
    cluster_profiles = torch.zeros((num_clusters, num_anatomical_rois), device=fiber_rois.device)

    for i in range(len(preds)):
        cluster_id = preds[i]
        roi_ids = fiber_rois[i].long()
        roi_hist = torch.bincount(roi_ids, minlength=num_anatomical_rois).float()
        roi_hist /= fiber_rois.shape[1]
        cluster_profiles[cluster_id] += roi_hist

    cluster_profiles /= cluster_profiles.sum(dim=1, keepdim=True).clamp(min=1e-6)
    cluster_profiles = (cluster_profiles >= threshold).float()

    return cluster_profiles

def compute_cluster_roi(X_train, y_train, num_of_class, threshold=1e-8):
    """
    Compute ROI classification for each cluster based on the rule that 40% of fibers pass through an ROI (ignoring 0).
    Also, print the ROI each fiber passes through along with its corresponding cluster.

    Parameters:
        X_train: Tensor, shape (b, 4, 100), where the 4th dimension represents ROI classification.
        y_train: Tensor, shape (b,), containing each fiber's cluster label.
        num_of_class: int, total number of clusters.
        threshold: float, threshold setting (default is 0.4, i.e., 40%).

    Returns:
        cluster_rois: List[Tensor], each element contains the ROI classification of the cluster (deduplicated & filtered by threshold).
    """
    # Extract fiber ROI classification information (b, 100)
    roi_data = X_train[:, 3, :]

    # Compute unique ROI classifications for each fiber (remove duplicates & ignore 0)
    fiber_rois = [torch.unique(roi[roi != 0]) for roi in roi_data]
    # Compute ROI for each cluster
    cluster_rois = []
    for cluster_id in range(num_of_class):
        # Get fibers belonging to the current cluster
        cluster_fibers = [fiber_rois[i] for i in range(len(y_train)) if y_train[i] == cluster_id]

        if not cluster_fibers:  # If no fibers belong to this cluster, skip
            cluster_rois.append(torch.tensor([]))
            continue

        # Aggregate all ROI classifications from fibers
        all_rois = torch.cat(cluster_fibers)  # Concatenate all fiber ROIs
        unique_rois, counts = torch.unique(all_rois, return_counts=True)  # Count occurrences of each ROI

        # Compute the occurrence ratio
        fiber_count = len(cluster_fibers)  # Total number of fibers in this cluster
        roi_ratio = counts.float() / fiber_count  # Compute the proportion of fibers passing through each ROI

        # Select ROIs that meet the 40% threshold
        selected_rois = unique_rois[roi_ratio >= threshold]
        cluster_rois.append(selected_rois)

    return cluster_rois

"""training settings"""
parser = argparse.ArgumentParser(description='naive CNN with weighted loss')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--patience', type=int, default=100, metavar='N',
                    help='(default: 100)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--center_weight', type=float, default=1.0, metavar='RT',
                    help='Center Loss Weight (default: 1.0)')
parser.add_argument('--kl_weight', type=float, default=2.0, metavar='RT',
                    help='KL Div. Distil Weight (default: 2.0)')
parser.add_argument('--feat_map_weight', type=float, default=2e-6, metavar='RT',
                    help='Feat Map Distil Weight (default: 2e-6)')
parser.add_argument('--kl_temp', type=int, default=2, metavar='RT',
                    help='KL Div. Temperature to smooth logits (default: 2)')
parser.add_argument('--use_only_main_loss', action='store_true', default=False,
                    help='Uses Only Main Loss between GT and output')
parser.add_argument('--use_main_plus_KL_Distil_loss', action='store_true', default=False,
                    help='Uses Main and KL-Div Distillation Loss between GT and intermediate outputs')
parser.add_argument('--use_main_plus_KL_plus_feature_Distil_loss', action='store_true', 
                    default=False, help='Uses Main, KL-Div and Feature Map Distillation Loss')
parser.add_argument('--use_clustering_loss', action='store_true', 
                    default=False, help='Uses Main, KL-Div, Feature Map Distillation and clustering Loss')
parser.add_argument('--clustering_weight', type=float, default=10, metavar='RT',
                    help='Clustering Weight (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--self_target_distribution', action='store_true', default=False,
                    help='disables self-training with target distribution')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 666)')
parser.add_argument('--use_dice_a_loss', action='store_true', 
                    default=False, help='Uses use_dice_a_loss')

parser.add_argument('--use_center_loss', action='store_true', 
                    default=False, help='Uses use_center_loss')

args = parser.parse_args()
# print(args.batch_size)
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
"""build datasets"""
with open('../data_61_26_ROIV2.pkl','rb') as f:
    data=pickle.load(f)

# raise ValueError('Please check the data file path')
    
"""flip and shuffle"""     
dataList=datato3d([data['X_train'],data['X_test']])
X_train=dataList[0] # no_tractsx4x100
X_test=dataList[1] # no_tractsx4x100

# check with a small dataset
# also test withh only 20 points per fiber: 1 point every 5 points
indices_train = np.random.choice(X_train.shape[0], size=500000, replace=False)
indices_test = np.random.choice(X_test.shape[0], size=50000, replace=False)

X_train, y_train = X_train[indices_train, :, ::], data['y_train'][indices_train]
X_test, y_test = X_test[indices_test, :, ::], data['y_test'][indices_test]

# Original Data - Many Fibers
# X_train, y_train = X_train[:, :, ::5], data['y_train'][:]
# X_test, y_test = X_test[:, :, ::5], data['y_test'][:]

# y_test_list=data['y_test'].tolist() # Original test set - many fibers
y_test_list=data['y_test'][indices_test].tolist() # select only "indices_test" no. of fibers for testing with small dataset

NCLASS=max(y_test_list)+1
num_anatomical_rois = 693
X_train,y_train=udflip(X_train,y_train,shuffle=True)
X_test,y_test=udflip(X_test,y_test,shuffle=False)

X_train=torch.from_numpy(X_train) # data['X_train'])
y_train=torch.from_numpy(y_train.astype(np.int32)) # data['y_train'])

X_test=torch.from_numpy(X_test)
y_test=torch.from_numpy(y_test.astype(np.int32))

del data,dataList
gc.collect()
print('data loaded!')
print('X_train_shape',X_train.size())
print('X_test_shape',X_test.size())

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
trn_set=utils.TensorDataset(X_train,y_train)
trn_loader=utils.DataLoader(trn_set,batch_size=args.batch_size,shuffle=True,**kwargs)

tst_set=utils.TensorDataset(X_test,y_test)
tst_loader=utils.DataLoader(tst_set,batch_size=args.test_batch_size,shuffle=False,**kwargs)

"""init model"""
model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS)
loss_nll = nn.NLLLoss(size_average=True) # log-softmax applied in the network
# init ROI cluster
# Center Loss
centerloss=CenterLoss(NCLASS,512,loss_weight=args.center_weight) # 512*4 if bottleneck applied

# KL-Div Loss
kl_loss=KLDivLoss(NCLASS, loss_weight=args.kl_weight, temperature=args.kl_temp)

# Clustering Loss
clustering_layer = ClusterlingLayer(embedding_dimension=512, num_clusters=NCLASS, alpha=1.0)

# IMP NOTE: The embedding dimension in both CenterLoss and ClusteringLayer is hardcoded to 512.
# If you change the embedding dimension in the model, you need to change it in the CenterLoss and ClusteringLayer as well. 

if args.cuda:
    model.cuda()
    loss_nll.cuda()
    kl_loss.cuda()
    centerloss.cuda()
    clustering_layer.cuda()

optimizer_nll = AdamW(model.parameters(),lr=args.lr)
optimizer_center = AdamW(centerloss.parameters(),lr=args.lr)
optimizer_cluster = AdamW(clustering_layer.parameters(), lr=args.lr)


def focalLoss(output,target):
    '''
    Args:
        y: (tensor) sized [N,].
    Return:
        (tensor) focal loss.
    '''
    alpha = 0.75
    gamma = 2
    logp = output
    p = logp.exp()
    w = alpha*(target>0).float() + (1-alpha)*(target==0).float()
    wp = w.view(-1,1) * (1-p).pow(gamma) * logp
    # TODO here we need to divided by batch size
    return loss_nll(wp,target.long()) / target.shape[0]

def feature_loss_function(fea, target_fea):
    """Compute L2 Loss between feature maps"""
    """
    param: fea: mid-level deep features.
    param: target_fea: deepest feature maps.
    """
    loss_feat = ((fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float())
    # TODO here we need to divided by batch size
    return args.feat_map_weight * torch.abs(loss_feat).sum() / fea.shape[0]


def compute_fiber_roi(fiber_data):
    """
    Compute the unique ROI classification for each fiber, removing ROI values of 0.

    Parameters:
        fiber_data: Tensor of shape (b, 4, 100), where the last dimension represents ROI classification.

    Returns:
        roi_list: List[Tensor], each element contains a fiber's unique ROI classifications (deduplicated, excluding 0).
    """
    # Extract ROI classification data (b, 100)
    roi_data = fiber_data[:, 3, :]

    # Remove 0 and get unique ROI classifications
    roi_list = [torch.unique(roi[roi != 0]) for roi in roi_data]
    return roi_list


def train(epoch):
    print(f'\nEpoch: {epoch}')
    model.train()
    training_loss = 0.0
    focal_loss = 0.0
    centering_loss=0.
    clustering_loss = 0.0
    preds = []
    labels = []
    print(f'args.use_clustering_loss: {args.use_clustering_loss}')
    print(f'args.use_dice_a_loss: {args.use_dice_a_loss}')
    print(f'args.use_center_loss: {args.use_center_loss}')
    print(f'args.clustering_weight: {args.clustering_weight}')
    global global_cluster_rois
    for batch_idx, (data, target) in enumerate(trn_loader):
        
        # print(f'\batch_idx: {batch_idx}')
        labels += target.numpy().tolist()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        
        # get 3d coordinate here for embed
        data_3d = data[:, 0:3, :]
        output, embed, _, _, _, _, _, _, _, _, _ = model(data_3d)
        predic_class = output.data.max(1, keepdim=True)[1]
        # print(f'predic_class: {predic_class}')
        # print(f'target class: {target}')
        # focal loss
        floss = focalLoss(output, target)

        # total loss at least should have focal loss
        total_loss = floss 
        if args.use_center_loss:
            closs = centerloss(target,embed)
            total_loss += closs

        if args.use_clustering_loss:
            if args.use_dice_a_loss:
                anatomical_info = compute_fiber_roi(data)
                # calculate clustering output using global cluster rois
                clustering_out, x_dis = clustering_layer(embed, anatomical_info=anatomical_info, cluster_rois=global_cluster_rois, predic=predic_class)
            else:
                clustering_out, x_dis = clustering_layer(embed)


            # calculate clustering loss（with or without dice）            
            # TODO need to double check
            tar_dist = ClusterlingLayer.create_soft_labels(target, NCLASS, temperature=args.kl_temp).to(target.device)

            
            loss_clust = args.clustering_weight * kl_loss.kl_div_cluster(torch.log(clustering_out), tar_dist) / args.batch_size

            total_loss += loss_clust
            clustering_loss += loss_clust.item()
        
        # record total training loss for this epoch    
        training_loss += total_loss.item()
        
        optimizer_nll.zero_grad()
        
        if args.use_center_loss:
            optimizer_center.zero_grad()
        if args.use_clustering_loss:
            optimizer_cluster.zero_grad()
            
        total_loss.backward()
        
        optimizer_nll.step()
        if args.use_center_loss:
            centering_loss+=closs.data
            optimizer_center.step()
        if args.use_clustering_loss:
            optimizer_cluster.step()

        pred = output.data.max(1, keepdim=True)[1]
        preds += pred.cpu().numpy().tolist()

    num_batch = len(trn_loader) / args.batch_size
    
    num_batch = len(trn_loader) / args.batch_size
    precision,recall,f1,sup=precision_recall_fscore_support(labels,preds,average='macro')
    avg_training_loss = training_loss / num_batch
    avg_clustering_loss = clustering_loss / num_batch if args.use_clustering_loss else 0.0
    print('\tCenter loss: {:.4f}'.format(centering_loss/num_batch))
    print(f'Training set avg loss: {avg_training_loss:.4f}')
    print(f'\tClustering loss: {avg_clustering_loss:.4f}' if args.use_clustering_loss else '')
    print('Precision,Recall,macro_f1',precision,recall,f1)
    return avg_training_loss
    

    
def test():
    """
    Evaluate the model on the test set.

    - Computes focal loss and optionally center loss and clustering loss.
    - Uses `global_cluster_rois` for anatomical consistency if `use_dice_a_loss` is enabled.
    - Updates `global_cluster_rois` using batch-level cluster anatomical profiles.
    """
    model.eval()
    test_loss = 0.0
    clustering_loss = 0.0
    centering_loss = 0.0
    probs = []
    preds = []
    labels = []

    global global_cluster_rois  # Ensure global access to cluster anatomical profiles

    with torch.no_grad():
        for data, target in tst_loader:
            labels += target.cpu().numpy().tolist()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # Extract 3D coordinates for embedding input
            data_3d = data[:, 0:3, :]
            output, embed, _, _, _, _, _, _, _, _, _ = model(data_3d)

            # Compute focal loss
            floss_main = focalLoss(output, target)
            total_loss = floss_main

            # Compute center loss if enabled
            if args.use_center_loss:
                centering_loss += centerloss(target.long(), embed).data

            # Compute clustering loss if enabled
            if args.use_clustering_loss:
                if args.use_dice_a_loss:
                    anatomical_info = compute_fiber_roi(data)
                    clustering_out, x_dis = clustering_layer(embed, anatomical_info=anatomical_info, cluster_rois=global_cluster_rois, predic=output)
                else:
                    clustering_out, x_dis = clustering_layer(embed)

                # Get predicted cluster labels
                tar_dist = ClusterlingLayer.create_soft_labels(target, NCLASS, temperature=args.kl_temp).to(target.device)
                loss_clust = args.clustering_weight * kl_loss.kl_div_cluster(torch.log(clustering_out), tar_dist) / args.batch_size

                total_loss += loss_clust
                clustering_loss += loss_clust.item()

            # Accumulate total test loss
            test_loss += total_loss.item()
            probs.append(output.data.cpu().numpy())

    # Compute final predictions using test-time augmentation
    preds = aug_at_test(probs, mode='max')
    num_batch = len(tst_loader) / args.batch_size

    # Compute evaluation metrics
    conf_mat = confusion_matrix(y_test_list, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_list, preds, average='macro')

    avg_testing_loss = test_loss / num_batch
    avg_clustering_loss = clustering_loss / num_batch if args.use_clustering_loss else 0.0

    print('\tCenter loss: {:.4f}'.format(centering_loss / num_batch))
    print(f'Test set avg loss: {avg_testing_loss:.4f}')
    if args.use_clustering_loss:
        print(f'\tClustering loss: {avg_clustering_loss:.4f}')
    print('Precision, Recall, macro F1:', precision, recall, f1)

    return avg_testing_loss, conf_mat, precision, recall, f1


    

"""start to train"""
best_epoch_idx=-1
best_f1=0.
history=list()
avg_training_loss_record=list()
avg_testing_loss_record=list()
patience=args.patience
if args.use_dice_a_loss:
    print(f'Creating cluster level roi profile')
    global_cluster_rois = compute_cluster_roi(X_train, y_train, NCLASS)
    print(device)
    global_cluster_rois = [rois.to(device) for rois in global_cluster_rois] 
    print(f'cluster level roi profile is created')
    # Print cluster-level ROI classification results
    print("\n===== Cluster-Level ROI Classification =====")
    for cluster_id, rois in enumerate(global_cluster_rois):
        print(f"Cluster {cluster_id}: size {len(rois.tolist())}")
for epoch in range(0,args.epochs):
    t0=time.time()
    avg_training_loss=train(epoch)
    avg_training_loss_record.append(avg_training_loss)
    print(time.time()-t0,'seconds')
    t1=time.time()
    avg_testing_loss,conf_mat, precision, recall, f1=test()
    avg_testing_loss_record.append(avg_testing_loss)
    print(time.time()-t1,'seconds')
    history.append((conf_mat, precision, recall, f1))
    if f1>best_f1:
        patience=args.patience
        best_f1=f1
        best_epoch_idx=epoch
        if args.use_clustering_loss and args.use_main_plus_KL_plus_feature_Distil_loss and args.self_target_distribution:
            torch.save(model.state_dict(),'Network_with_clustering_loss_Self_Target.model')
        elif args.use_clustering_loss and args.use_main_plus_KL_plus_feature_Distil_loss and not args.self_target_distribution:
            torch.save(model.state_dict(),'Network_with_clustering_loss_Label_Target.model')
        elif not args.use_clustering_loss and args.use_main_plus_KL_plus_feature_Distil_loss and not args.self_target_distribution:
            torch.save(model.state_dict(),'Network_with_KL_loss_NO_Clustering.model')
        elif not args.use_clustering_loss and not args.use_main_plus_KL_plus_feature_Distil_loss and not args.self_target_distribution:
            torch.save(model.state_dict(),'Network_with_NO_Distillation_NO_Clustering.model')
        else:
            torch.save(model.state_dict(),'Name_your_network_accordingly.model')
    else:
        patience-=1
        if patience==0:
            break

print('Best epoch:{}\n'.format(best_epoch_idx))
conf_mat, precision, recall, f1=history[best_epoch_idx]
# print('conf_mat:\n',np.array_str(conf_mat,max_line_width=10000))
print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\n'.format(precision,recall,f1)) 
# np.save('confMat.npy',conf_mat)
loss_record=np.vstack((np.array(avg_training_loss_record),np.array(avg_testing_loss_record)))
# np.save('loss_record.npy',loss_record)
