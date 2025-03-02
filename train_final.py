#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19, 2024

Modified by Bohan on June 20, 2024
Tract Classification Project
- Added KL-divergence for Self-Distillation and clustering loss
- Added klDic.py and clustering_layer.py and modified main_40_16_GPU_KL.py file
- Updated RESNET152_ATT_naive.py file to include bottleneck layers for Self-Distillation (both at class level and feature-map level)

@author: Bohan
"""
from Embedding_layer import ROIFeatureExtractor
import RESNET152_ATT_naive
from CenterLoss import CenterLoss
from klDiv import KLDivLoss
from clustering_layer_v2 import ClusterlingLayer
from ADAMW import AdamW
from Util import *

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


def train_step(epoch, model, args, roi_extractor=None, roi_embedding_layer=None, device='cpu'):
    print(f'\nEpoch: {epoch}')
    model.train()
    log_training_total_loss = 0.0
    log_focal_loss = 0.0
    log_centering_loss= 0.
    log_clustering_loss = 0.0
    preds = []
    labels = []
    global global_cluster_rois
    for batch_idx, (data, target) in enumerate(trn_loader):

        labels += target.numpy().tolist()

        data, target = data.to(device), target.to(device)
        data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
        # get 3d coordinate here for embed, here we use 3D + ROI embed
        if args.use_feature_extractor:
            data_processed = preprocess_fiber_input(data, roi_extractor=roi_extractor, device=device, net_type='FE')
        elif args.use_embedding:
            data_processed = preprocess_fiber_input(data, roi_embedding_layer=roi_embedding_layer, device=device, net_type='EB')
        elif args.use_concat:
            data_processed = preprocess_fiber_input(data, device=device, net_type='concat')
        else:
            data_processed = preprocess_fiber_input(data, device=device) 

        output, embed, _, _, _, _, _, _, _, _, _ = model(data_processed)
        predic_class = output.data.max(1, keepdim=True)[1]

        # focal loss
        floss = focalLoss(output, target, loss_nll)
        # total loss at least should have focal loss
        total_loss = floss
        log_focal_loss += floss.item()

        if args.use_center_loss:
            closs = centerloss(target,embed)
            total_loss += closs
            log_centering_loss += closs.item()

        if args.use_clustering_loss:
            if args.use_dice_a_loss:
                anatomical_info = compute_fiber_roi(data)
                # calculate clustering output using global cluster rois
                clustering_out, x_dis = clustering_layer(embed, anatomical_info=anatomical_info, cluster_rois=global_cluster_rois, predic=predic_class)
            else:
                clustering_out, x_dis = clustering_layer(embed)

            
            # TODO need to double check
            tar_dist = ClusterlingLayer.create_soft_labels(target, NCLASS, temperature=args.kl_temp).to(target.device)

            
            clust_loss = args.clustering_weight * kl_loss.kl_div_cluster(torch.log(clustering_out), tar_dist) / args.batch_size

            total_loss += clust_loss
            log_clustering_loss += clust_loss.item()
        
        # record total training loss for this epoch    
        log_training_total_loss += total_loss.item()
        
        optimizer_nll.zero_grad()
        
        if args.use_center_loss:
            optimizer_center.zero_grad()
        if args.use_clustering_loss:
            optimizer_cluster.zero_grad()
        if args.use_feature_extractor:
            optimizer_fe.zero_grad()
        if args.use_embedding:
            optimizer_eb.zero_grad()     
          
        total_loss.backward()
        
        optimizer_nll.step()
        if args.use_center_loss:
            optimizer_center.step()
        if args.use_clustering_loss:
            optimizer_cluster.step()
        if args.use_feature_extractor:
            optimizer_fe.step()
        if args.use_embedding:
            optimizer_eb.step()      

        pred = output.data.max(1, keepdim=True)[1]
        preds += pred.cpu().numpy().tolist()

    
    # log information
    num_batch = len(trn_loader) / args.batch_size
    precision,recall,f1,sup=precision_recall_fscore_support(labels, preds, average='macro')
    avg_training_loss = log_training_total_loss / num_batch
    print(f'Training set avg loss: {avg_training_loss:.4f}')
    print('\tfocal loss: {:.4f}'.format(log_focal_loss/num_batch))
    if args.use_center_loss:
        print('\tCenter loss: {:.4f}'.format(log_centering_loss/num_batch))
    if args.use_clustering_loss:    
        avg_clustering_loss = log_clustering_loss / num_batch
        print(f'\tClustering loss: {avg_clustering_loss:.4f}' if args.use_clustering_loss else '')
    print('Precision, Recall, macro_f1', precision, recall, f1)    
    return avg_training_loss
    
def test(epoch, model, args, roi_extractor=None, roi_embedding_layer=None, device='cpu'):
    """
    Evaluate the model on the test set.

    - Computes focal loss and optionally center loss and clustering loss.
    - Uses `global_cluster_rois` for anatomical consistency if `use_dice_a_loss` is enabled.
    - Updates `global_cluster_rois` using batch-level cluster anatomical profiles.
    """
    model.eval()
    log_testing_total_loss = 0.0
    log_focal_loss = 0.0
    log_centering_loss= 0.
    log_clustering_loss = 0.0
    probs = []
    preds = []
    labels = []

    global global_cluster_rois  # Ensure global access to cluster anatomical profiles

    with torch.no_grad():
        for data, target in tst_loader:
            labels += target.cpu().numpy().tolist()
            # if args.cuda:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)

            # Extract 3D coordinates for embedding input
            if args.use_feature_extractor:
                data_processed = preprocess_fiber_input(data, roi_extractor=roi_extractor, device=device, net_type='FE')
            elif args.use_embedding:
                data_processed = preprocess_fiber_input(data, roi_embedding_layer=roi_embedding_layer, device=device, net_type='EB')
            elif args.use_concat:
                data_processed = preprocess_fiber_input(data, device=device, net_type='concat')
            else:
                data_processed = preprocess_fiber_input(data, device=device) 
            output, embed, _, _, _, _, _, _, _, _, _ = model(data_processed)

            # Compute focal loss
            floss = focalLoss(output, target, loss_nll=loss_nll)
            total_loss = floss
            log_focal_loss += floss.item()
            # Compute center loss if enabled
            if args.use_center_loss:
                log_centering_loss += centerloss(target.long(), embed).item()

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
                log_clustering_loss += loss_clust.item()

            # Accumulate total test loss
            log_testing_total_loss += total_loss.item()
            probs.append(output.data.cpu().numpy())

    # Compute final predictions using test-time augmentation
    preds = aug_at_test(probs, mode='max')
    num_batch = len(tst_loader) / args.batch_size

    # Compute evaluation metrics
    conf_mat = confusion_matrix(y_test_list, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_list, preds, average='macro')

    avg_testing_loss = log_testing_total_loss / num_batch
    avg_clustering_loss = log_clustering_loss / num_batch
    print('\tCenter loss: {:.4f}'.format(log_centering_loss / num_batch))
    print('\tfocal loss: {:.4f}'.format(log_focal_loss/num_batch))
    print(f'Test set avg loss: {avg_testing_loss:.4f}')
    if args.use_clustering_loss:
        print(f'\tClustering loss: {avg_clustering_loss:.4f}')
    print('Precision, Recall, macro F1:', precision, recall, f1)

    return avg_testing_loss, conf_mat, precision, recall, f1


    
if __name__ == "__main__":
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

    parser.add_argument('--quick_test', action='store_true', 
                        default=False, help='do a quick testing with a small set')

    parser.add_argument('--use_feature_extractor', action='store_true', 
                        default=False, help='use FE for ROI data')

    parser.add_argument('--use_embedding', action='store_true', 
                        default=False, help='use embding for ROI data')

    parser.add_argument('--use_concat', action='store_true', 
                        default=False, help='use directly concat for ROI data')
    parser.add_argument('--num_roi', type=int, default=726, metavar='RT',
                        help='number of ROI classes')
    parser.add_argument('--emb_dim', type=int, default=32, metavar='RT',
                        help='number of embedding dim')
    parser.add_argument('--device', type=str, default='0', metavar='RT',
                        help='device num')
    args = parser.parse_args()

    print(f'args.use_clustering_loss: {args.use_clustering_loss}')
    print(f'args.use_dice_a_loss: {args.use_dice_a_loss}')
    print(f'args.use_center_loss: {args.use_center_loss}')
    print(f'args.clustering_weight: {args.clustering_weight}')


    # print(args.batch_size)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Hyperparameters configuration
    HIDDEN_DIM = 64
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    #TODO check use case
    GPUINX=args.device
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=GPUINX
    np.random.seed(987)
    device = torch.device("cuda:{}".format(GPUINX) if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')   
    """build datasets"""
    with open('../data_47_20_ROI_Final.pkl','rb') as f:
        data=pickle.load(f)
    # with open('../data_47_20_ROI_Final_0.2downsample.pkl','rb') as f:
    #     data=pickle.load(f)

    # raise ValueError('Please check the data file path')
        
    """flip and shuffle"""     
    dataList = datato3d([data['X_train'],data['X_test']])
    X_train=dataList[0] # no_tractsx4x100
    X_test=dataList[1] # no_tractsx4x100


    if args.quick_test:
        # quick testing with a small dataset
        # also test withh only 20 points per fiber: 1 point every 5 points
        indices_train = np.random.choice(X_train.shape[0], size=500000, replace=False)
        indices_test = np.random.choice(X_test.shape[0], size=50000, replace=False)
        X_train, Y_train = X_train[indices_train, :, ::], data['y_train'][indices_train]
        X_test, Y_test = X_test[indices_test, :, ::], data['y_test'][indices_test]
        y_test_list=data['y_test'][indices_test].tolist()
    else: 
        # TODO full testing
        X_train, Y_train = X_train, data['y_train']
        X_test, Y_test = X_test, data['y_test']
        y_test_list=data['y_test'].tolist() # select only "indices_test" no. of fibers for testing with small dataset

    NCLASS = max(y_test_list) + 1
    # data augmentation
    X_train, Y_train = udflip(X_train, Y_train, shuffle=True)
    X_test, Y_test = udflip(X_test, Y_test, shuffle=False)

    X_train=torch.from_numpy(X_train) # data['X_train'])
    Y_train=torch.from_numpy(Y_train.astype(np.int32)) # data['y_train'])

    X_test=torch.from_numpy(X_test)
    Y_test=torch.from_numpy(Y_test.astype(np.int32))

    del data,dataList
    gc.collect()
    print('data loaded!')
    print('X_train_shape',X_train.size())
    print('X_test_shape',X_test.size())


    trn_set = utils.TensorDataset(X_train,Y_train)
    trn_loader = utils.DataLoader(trn_set,batch_size=args.batch_size,shuffle=True,**kwargs)

    tst_set=utils.TensorDataset(X_test,Y_test)
    tst_loader=utils.DataLoader(tst_set,batch_size=args.test_batch_size,shuffle=False,**kwargs)
    roi_embedding_layer = None
    roi_extractor = None
    """init model"""
    if args.use_feature_extractor:
        ROI_EMBEDDING_DIM = args.emb_dim
        NUM_ROI_CLASSES = args.num_roi + 1
        model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=3+ROI_EMBEDDING_DIM)
        # init ROI Embedding layer
        roi_embedding_layer = nn.Embedding(NUM_ROI_CLASSES, ROI_EMBEDDING_DIM).to(device)
        # init FE
        roi_extractor = ROIFeatureExtractor(roi_embedding_layer, ROI_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)
        roi_extractor.to(device)
    elif args.use_embedding:
        ROI_EMBEDDING_DIM = args.emb_dim
        NUM_ROI_CLASSES = args.num_roi + 1
        # init ROI Embedding layer
        roi_embedding_layer = nn.Embedding(NUM_ROI_CLASSES, ROI_EMBEDDING_DIM).to(device)
        model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=3+ROI_EMBEDDING_DIM)
    elif args.use_concat:
        model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=4)
    
    else:
        print("3D only")
        model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=3)  


    # by defaul focal loss is used
    loss_nll = nn.NLLLoss(size_average=True) # log-softmax applied in the network

    # init ROI cluster
    # Center Loss
    if args.use_center_loss:
        centerloss = CenterLoss(NCLASS,512,loss_weight=args.center_weight) # 512*4 if bottleneck applied
        centerloss.to(device)
    if args.use_clustering_loss:
        # KL-Div Loss
        kl_loss = KLDivLoss(NCLASS, loss_weight=args.kl_weight, temperature=args.kl_temp)
        # Clustering Loss
        clustering_layer = ClusterlingLayer(embedding_dimension=512, num_clusters=NCLASS, alpha=1.0)
        kl_loss.to(device)
        clustering_layer.to(device)

    # IMP NOTE: The embedding dimension in both CenterLoss and ClusteringLayer is hardcoded to 512.
    # If you change the embedding dimension in the model, you need to change it in the CenterLoss and ClusteringLayer as well. 
    if args.cuda:
        model.to(device)
        loss_nll.to(device)

    optimizer_nll = AdamW(model.parameters(),lr=args.lr)
    if args.use_center_loss:
        optimizer_center = AdamW(centerloss.parameters(),lr=args.lr)
    if args.use_clustering_loss:    
        optimizer_cluster = AdamW(clustering_layer.parameters(), lr=args.lr)
    if args.use_feature_extractor:
        optimizer_fe = AdamW(roi_extractor.parameters(), lr=args.lr)
    if args.use_embedding:    
        optimizer_eb = AdamW(roi_embedding_layer.parameters(), lr=args.lr)
    """start to train"""
    best_epoch_idx = -1
    best_f1=0.
    history=list()
    avg_training_loss_record=list()
    avg_testing_loss_record=list()
    patience=args.patience
    print(args)
    if args.use_dice_a_loss:
        print(f'Creating cluster level roi profile')
        global_cluster_rois = compute_cluster_roi(X_train, Y_train, NCLASS)
        global_cluster_rois = [rois.to(device) for rois in global_cluster_rois] 
        print(f'cluster level roi profile is created')
        # Print cluster-level ROI classification results
        print("\n===== Cluster-Level ROI Classification =====")
        for cluster_id, rois in enumerate(global_cluster_rois):
            print(f"Cluster {cluster_id}: size {len(rois.tolist())}")

    network_design = ''
    loss_design = 'focal_loss_'
    if args.use_feature_extractor:
        network_design = f'FE_dim_{args.emb_dim}'
    elif args.use_embedding:
        network_design = f'EB_dim_{args.emb_dim}'
    elif args.use_concat:
        network_design = 'concat'
    else:
        network_design =  'no_roi'
                
    if args.use_clustering_loss and args.use_dice_a_loss:
        loss_design += f'and_dice_cluster_loss_c_{args.clustering_weight}_'
    elif args.use_clustering_loss:
        loss_design += f'and_cluster_loss_c_{args.clustering_weight}_'

    if args.use_center_loss:
        loss_design += 'and_center_loss_'

    model_saved_name = f'{loss_design}{network_design}.model'
    print(f'saved name: {model_saved_name}')

    for epoch in range(0,args.epochs):
        t0=time.time()
        avg_training_loss=train_step(epoch, args=args, model=model, roi_embedding_layer=roi_embedding_layer, roi_extractor=roi_extractor, device=device)
        avg_training_loss_record.append(avg_training_loss)
        print(time.time()-t0,'seconds')
        t1=time.time()
        avg_testing_loss,conf_mat, precision, recall, f1=test(epoch, args=args, model=model, roi_embedding_layer=roi_embedding_layer, roi_extractor=roi_extractor, device=device)
        avg_testing_loss_record.append(avg_testing_loss)
        print(time.time()-t1,'seconds')
        history.append((conf_mat, precision, recall, f1))
        if f1>best_f1:
            patience=args.patience
            best_f1=f1
            best_epoch_idx=epoch
  
            torch.save(model.state_dict(),model_saved_name)
            if args.use_feature_extractor:
                torch.save(roi_extractor.state_dict(),'FE_layer_' + model_saved_name)
            elif args.use_embedding:
                torch.save(roi_embedding_layer.state_dict(),'EB_layer_' + model_saved_name)
            if args.use_clustering_loss:
                torch.save(clustering_layer.state_dict(),'CLS_layer_' + model_saved_name)
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
