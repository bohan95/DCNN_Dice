#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 01:25:02 2018

@author: ht
"""
import RESNET152_ATT_naive
from CenterLoss import CenterLoss
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

#"""confusion matrix"""
#def getConfusionInfo(mat,classname):#input: np array,list of classnames
#    for i in range(mat.shape[0]):
#        mat[i]=mat[i]/np.sum(mat[i])
#    df_cm = pd.DataFrame(mat, index = classname,columns = classname)
#    plt.figure(figsize = (10,7))
#    sn.heatmap(df_cm, annot=True, fmt='.2f')
#    
#    for i in range(mat.shape[0]):
#        tp=mat[i,i]
#        fp=np.sum(mat[:,i])-mat[i,i]
#        fn=np.sum(mat[i,:])-mat[i,i]
#        if 2*tp+fn+fp==0:
#            f1=0.
#        else:
#            f1=2*tp/(2*tp+fn+fp)
#        print(classname[i],f1)
        
"""v flip and shuffle"""
def udflip(X_nparray,y_nparray,shuffle=True):
    output=np.zeros((X_nparray.shape[0],X_nparray.shape[1],X_nparray.shape[2]),dtype=np.float32)
    for i in range(X_nparray.shape[0]):
        output[i,:]=np.flipud(X_nparray[i,:])
    output=np.vstack((X_nparray,output))
    y=np.hstack((y_nparray,y_nparray))
    if shuffle:
        shuffle_inx=np.random.permutation(output.shape[0])
        return output[shuffle_inx],y[shuffle_inx]
    else:
        return output,y


def aug_at_test(probs,mode='max'):
    #input: list of numpy 10000*18 (logits)
    #output: final decisions
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

def datato3d(arrays):#list of np arrays, NULL*3*100
    output=list()
    for i in arrays:
        i=np.transpose(i,(0,2,1))
        output.append(i)
    return output

"""training settings"""
parser = argparse.ArgumentParser(description='naive CNN with weighted loss')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--patience', type=int, default=15, metavar='N',
                    help='(default: 15)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--center_weight', type=float, default=1.0, metavar='RT',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
"""build datasets"""
with open('../preprocessing/data_28_14.pkl','rb') as f:
    data=pickle.load(f)

"""flip and shuffle"""
#label list
#i=0
#with open('label_list.csv','w') as f:
#    for l in data['classes']:
#        class_info=re.search('\d+-\d+',l)
#        f.write(class_info.group()+','+str(i)+'\n')
#        i+=1
#reshape        
dataList=datato3d([data['X_train'],data['X_test']])
X_train=dataList[0]
X_test=dataList[1]

X_train,y_train=udflip(X_train,data['y_train'],shuffle=True)
X_test,y_test=udflip(X_test,data['y_test'],shuffle=False)

X_train=torch.from_numpy(X_train)#data['X_train'])
y_train=torch.from_numpy(y_train.astype(np.int32))#data['y_train'])

X_test=torch.from_numpy(X_test)
y_test=torch.from_numpy(y_test.astype(np.int32))

y_test_list=data['y_test'].tolist()

NCLASS=max(y_test_list)+1

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
model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS)###RESNET152.resnet152(num_classes=max(y_test_list)+1)
###load weight
##model.load_state_dict(torch.load('../../code/RESNET_system6_ATT/best.model'))
##print('weights loaded!')
#########
loss = nn.NLLLoss(size_average=True)#log-softmax applied in the network
centerloss=CenterLoss(NCLASS,512,args.center_weight)#512*4 if bottleneck applied
if args.cuda:
    model.cuda()
    loss.cuda()
    centerloss.cuda()
optimizer_nll = AdamW(model.parameters(),lr=args.lr)###torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer_center = AdamW(centerloss.parameters(),lr=1e-4)###torch.optim.Adam(centerloss.parameters(), lr=0.0001)
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
    return loss(wp,target)
def train(epoch):
    print('\n\nEpoch: {}'.format(epoch))
    model.train()
    training_loss=0.
    centering_loss=0.
    preds=list()
    labels=list()
    for batch_idx,(data,target) in enumerate(trn_loader):
        labels+=target.numpy().tolist()
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        
        output,embed,_=model(data)
        tloss=focalLoss(output,target)
        closs=centerloss(target,embed)
        totalloss=tloss+closs
        ###print(tloss.data[0])
        optimizer_nll.zero_grad()
        optimizer_center.zero_grad()
        
        totalloss.backward()
        
        training_loss+=tloss.data[0]
        centering_loss+=closs.data[0]
        
        optimizer_nll.step()
        optimizer_center.step()
        
        pred = output.data.max(1, keepdim=True)[1]
        preds+=pred.cpu().numpy().tolist()
        
    conf_mat=confusion_matrix(labels,preds)
    precision,recall,f1,sup=precision_recall_fscore_support(labels,preds,average='macro')
    avg_training_loss=training_loss/len(trn_loader)
    print('Training set avg loss: {:.4f}'.format(avg_training_loss))
    print('\tcenter loss: {:.4f}'.format(centering_loss/len(trn_loader)))
    print('conf_mat:\n',np.array_str(conf_mat,max_line_width=10000))
    print('Precision,Recall,macro_f1',precision,recall,f1)
    return avg_training_loss
    

    
def test():
    model.eval()
    test_loss = 0
    centering_loss=0.
    #preds=list()
    probs=list()
    for data, target in tst_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output,embed,_ = model(data)
        ###test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        test_loss+=loss(output,target).data[0]
        centering_loss+=centerloss(target,embed).data[0]
        ###print(test_loss)
        probs.append(output.data.cpu().numpy())
    
    ###test_loss /= len(tst_loader.dataset)
    preds=aug_at_test(probs,mode='max')
    conf_mat=confusion_matrix(y_test_list,preds)
    precision,recall,f1,sup=precision_recall_fscore_support(y_test_list,preds,average='macro')
    avg_testing_loss=test_loss/len(tst_loader)
    print('Test set avg loss: {:.4f}'.format(avg_testing_loss))
    print('\tcenter loss: {:.4f}'.format(centering_loss/len(tst_loader)))
    print('conf_mat:\n',np.array_str(conf_mat,max_line_width=10000))
    print('Precision,Recall,macro_f1',precision,recall,f1)
    return avg_testing_loss,conf_mat,precision, recall, f1    
    

"""start to train"""
best_epoch_idx=-1
best_f1=0.
#f1_history=np.zeros(args.epochs,dtype=np.float32)
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
        torch.save(model.state_dict(),'center_focal_attnaive.model.'+GPUINX)
    else:
        patience-=1
        if patience==0:
            break

print('Best epoch:{}\n'.format(best_epoch_idx))
conf_mat, precision, recall, f1=history[best_epoch_idx]
print('conf_mat:\n',np.array_str(conf_mat,max_line_width=10000))
print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\n'.format(precision,recall,f1)) 
np.save('confMat'+GPUINX+'.npy',conf_mat)
loss_record=np.vstack((np.array(avg_training_loss_record),np.array(avg_testing_loss_record)))
np.save('loss_record'+GPUINX+'.npy',loss_record)
#classname=['c'+str(i+1) for i in range(NCLASS)]
#getConfusionInfo(conf_mat,classname)
    
    
    
    
    
    
    
    
