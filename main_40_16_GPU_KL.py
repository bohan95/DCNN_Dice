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
from clustering_layer import ClusterlingLayer
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

GPUINX='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=GPUINX
np.random.seed(987)
        
"""v flip and shuffle"""
def udflip(X_nparray,y_nparray,shuffle=True):
    output=np.zeros((X_nparray.shape[0],X_nparray.shape[1],X_nparray.shape[2]),dtype=np.float32)
    for i in range(X_nparray.shape[0]):
        # a = X_nparray[i,:] # just for checking the code
        output[i,:]=np.flipud(X_nparray[i,:])
    output=np.vstack((X_nparray,output))
    y=np.hstack((y_nparray,y_nparray))
    if shuffle:
        shuffle_inx=np.random.permutation(output.shape[0])
        return output[shuffle_inx],y[shuffle_inx]
    else:
        return output,y


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
parser.add_argument('--clustering_weight', type=float, default=0.1, metavar='RT',
                    help='Clustering Weight (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--self_target_distribution', action='store_true', default=False,
                    help='disables self-training with target distribution')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 666)')

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

X_train, y_train = X_train[indices_train, :, ::5], data['y_train'][indices_train]
X_test, y_test = X_test[indices_test, :, ::5], data['y_test'][indices_test]

# Original Data - Many Fibers
# X_train, y_train = X_train[:, :, ::5], data['y_train'][:]
# X_test, y_test = X_test[:, :, ::5], data['y_test'][:]

# y_test_list=data['y_test'].tolist() # Original test set - many fibers
y_test_list=data['y_test'][indices_test].tolist() # select only "indices_test" no. of fibers for testing with small dataset

NCLASS=max(y_test_list)+1

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

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
trn_set=utils.TensorDataset(X_train,y_train)
trn_loader=utils.DataLoader(trn_set,batch_size=args.batch_size,shuffle=True,**kwargs)

tst_set=utils.TensorDataset(X_test,y_test)
tst_loader=utils.DataLoader(tst_set,batch_size=args.test_batch_size,shuffle=False,**kwargs)

"""init model"""
model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS)
loss_nll = nn.NLLLoss(size_average=True) # log-softmax applied in the network

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
    print("{:.10f}".format(p[0].item()))
    return loss_nll(wp,target.long())

def feature_loss_function(fea, target_fea):
    """Compute L2 Loss between feature maps"""
    """
    param: fea: mid-level deep features.
    param: target_fea: deepest feature maps.
    """
    loss_feat = ((fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float())
    return args.feat_map_weight * torch.abs(loss_feat).sum()

def train(epoch):
    print('\n\nEpoch: {}'.format(epoch))
    model.train()
    training_loss=0.
    centering_loss=0.
    clustering_loss=0.
    preds=list()
    labels=list()
    for batch_idx,(data,target) in enumerate(trn_loader):
        labels+=target.numpy().tolist()
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        
        # output,embed,_ = model(data) # Original return statement
        output,embed,_,non_softmax_out, out1, out2, out3, final_feat, out1_feat, out2_feat, out3_feat = model(data) # return statement with intermediate outputs for kl-divergence and feature-map loss      
        print(f"output shape: {output.shape}")
        print(f'target: {target}')
        # compute focal losses
        floss_main = focalLoss(output,target) # Main output loss from GT to output

        # compute focal losses for intermediate outputs (Self-Distillation from GT to intermediate outputs)
        floss_out1 = focalLoss(F.log_softmax(out1),target) 
        floss_out2 = focalLoss(F.log_softmax(out2),target)
        floss_out3 = focalLoss(F.log_softmax(out3),target)

        # compute KL-divergence losses
        # output logits before the log-softmax in the network
        non_softmax_out_temp = non_softmax_out / args.kl_temp
        softmax_out_temp = torch.softmax(non_softmax_out_temp, dim=1)

        kl_loss1 = kl_loss(out1, softmax_out_temp.detach()) 
        kl_loss2 = kl_loss(out2, softmax_out_temp.detach()) 
        kl_loss3 = kl_loss(out3, softmax_out_temp.detach()) 

        # Feature-Map Losses
        feature_loss_1 = feature_loss_function(out1_feat, final_feat.detach())
        feature_loss_2 = feature_loss_function(out2_feat, final_feat.detach())
        feature_loss_3 = feature_loss_function(out3_feat, final_feat.detach())

        # selection of losses
        if args.use_only_main_loss:
            tloss = floss_main # only main loss
            # print('Using only main loss')
        elif args.use_main_plus_KL_Distil_loss:
            tloss = floss_main + (floss_out1 + floss_out2 + floss_out3) + (kl_loss1 + kl_loss2 + kl_loss3)
            # print('Using main loss + KL-Div Distillation loss')
        elif args.use_main_plus_KL_plus_feature_Distil_loss:
            # total loss = main loss + KL-divergence loss + feature-map loss
            tloss = floss_main + (floss_out1 + floss_out2 + floss_out3) + (kl_loss1 + kl_loss2 + kl_loss3) + (feature_loss_1 + feature_loss_2 + feature_loss_3)
            # print('Using main loss + KL-Div Distillation loss + Feature-Map loss')
        else:
            raise ValueError('Please select a valid main loss function')

        # compute center losses
        closs=centerloss(target,embed)
        # print(f'tloss: {tloss.data[0]}, closs: {closs.data[0]}')
        totalloss=tloss+closs

        if args.use_clustering_loss:
            # compute clustering output probabilities
            clustering_out, x_dis = clustering_layer(embed)

            if args.self_target_distribution:
                # generate target distribution for self-learning as in the original paper
                tar_dist = ClusterlingLayer.target_distribution(clustering_out)
            else:
                # generate target distribution from ground-truth labels as target
                tar_dist = ClusterlingLayer.create_soft_labels(target, NCLASS, temperature=args.kl_temp)

            tar_dist = tar_dist
            # calculate the clustering loss
            loss_clust = args.clustering_weight * kl_loss.kl_div_cluster(torch.log(clustering_out), tar_dist) / args.batch_size
        
            totalloss += loss_clust
            # print('Using clustering loss')
            optimizer_cluster.zero_grad()

        ###print(tloss.data[0])
        optimizer_nll.zero_grad()
        optimizer_center.zero_grad()
        
        totalloss.backward()
        
        training_loss+=tloss.data
        centering_loss+=closs.data
        if args.use_clustering_loss:
            clustering_loss+=loss_clust.data
        
        optimizer_nll.step()
        optimizer_center.step()

        if args.use_clustering_loss:
            optimizer_cluster.step()
        
        pred = output.data.max(1, keepdim=True)[1]
        preds+=pred.cpu().numpy().tolist()
        
    conf_mat=confusion_matrix(labels,preds)
    precision,recall,f1,sup=precision_recall_fscore_support(labels,preds,average='macro')
    avg_training_loss=training_loss/len(trn_loader)
    print('Training set avg loss: {:.4f}'.format(avg_training_loss))
    print('\tCenter loss: {:.4f}'.format(centering_loss/len(trn_loader)))
    if args.use_clustering_loss:
        print('\tClustering loss: {:.4f}'.format(clustering_loss/len(trn_loader)))
    # print('conf_mat:\n',np.array_str(conf_mat,max_line_width=10000))
    print('Precision,Recall,macro_f1',precision,recall,f1)
    return avg_training_loss
    

    
def test():
    model.eval()
    test_loss = 0
    centering_loss=0.
    clustering_loss=0.
    probs=list()
    with torch.no_grad():
        for data, target in tst_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # output,embed,_ = model(data) # Original return statement
            output,embed,_,_, _, _, _, _,_,_,_ = model(data) # return statement with intermediate outputs for kl-divergence
        ###test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            test_loss+=loss_nll(output,target.long()).data
            centering_loss+=centerloss(target.long(),embed).data
            
            if args.use_clustering_loss:
                # compute clustering output probabilities
                clustering_out, x_dis = clustering_layer(embed)

                if args.self_target_distribution:
                    tar_dist = ClusterlingLayer.target_distribution(clustering_out)
                else:
                    # generate target distribution from one-hot ground-truth labels
                    tar_dist = ClusterlingLayer.create_soft_labels(target, NCLASS, temperature=args.kl_temp)

                tar_dist = tar_dist
                # calculate the clustering loss
                loss_clust = args.clustering_weight * kl_loss.kl_div_cluster(torch.log(clustering_out), tar_dist) / args.batch_size
            
                clustering_loss += loss_clust
                
        ###print(test_loss)
            probs.append(output.data.cpu().numpy())
    
    preds=aug_at_test(probs,mode='max')
    conf_mat=confusion_matrix(y_test_list,preds)
    precision,recall,f1,sup=precision_recall_fscore_support(y_test_list,preds,average='macro')
    avg_testing_loss=test_loss/len(tst_loader)
    print('Test set avg loss: {:.4f}'.format(avg_testing_loss))
    print('\tCenter loss: {:.4f}'.format(centering_loss/len(tst_loader)))
    if args.use_clustering_loss:
        print('\tClustering loss: {:.4f}'.format(clustering_loss/len(tst_loader)))
    # print('conf_mat:\n',np.array_str(conf_mat,max_line_width=10000))
    print('Precision,Recall,macro_f1',precision,recall,f1)
    return avg_testing_loss,conf_mat,precision, recall, f1    
    

"""start to train"""
best_epoch_idx=-1
best_f1=0.
history=list()
avg_training_loss_record=list()
avg_testing_loss_record=list()
patience=args.patience
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
