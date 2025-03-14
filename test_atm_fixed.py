import os
import sys
import gc
import torch
import numpy as np
import h5py
import scipy.io as spio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score
)

# Custom module imports
from Embedding_layer import ROIFeatureExtractor
import RESNET152_ATT_naive
from Util import focalLoss, preprocess_fiber_input
from clustering_layer_v2 import ClusterlingLayer
from klDiv import KLDivLoss

def loadmat(filename):
    output = dict()
    
    with h5py.File(filename, 'r') as data:
        if 'Whole_tracks' not in data:
            raise KeyError("❌ Error: 'Whole_tracks' doen't exist！")

        whole_tracks = data['Whole_tracks']  # 结构体 Whole_tracks

        if 'count' not in whole_tracks or 'data' not in whole_tracks:
            raise KeyError(f"❌ Error: 'Whole_tracks' incomplete, included:: {list(whole_tracks.keys())}")

        count = whole_tracks['count'][()]  
        print("🔍 Whole_tracks['count'] data:", count)
        print("🔍 type:", type(count))

        total_count = int(count.item())
        print(f'total_count: {total_count}')
        track = []
        for i in range(total_count):
            data_ref = whole_tracks['data'][i].item()
            track.append(np.transpose(data[data_ref][:]).astype(np.float32))

        output['tracks'] = {
            'count': total_count,
            'data': track
        }
    
    return output

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#%%
def mySoftmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div
"""normalize"""#110
def rescale(X_list,count):
    output=list()
    if count==1:
        output.append(X_list/110)
        return output
    for i in range(len(X_list)):
        output.append(X_list[i]/110)
    return output

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
def datato3d(arrays):#list of np arrays, NULL*3*100
    output=list()
    for i in arrays:
        i=np.squeeze(i,axis=1)
        i=np.transpose(i,(0,2,1))
        output.append(i)
    return output
def udflip(X_nparray, y_nparray, shuffle=True):

    if X_nparray.shape[2] == 4:
        if np.std(X_nparray[:, 0, :]) > np.std(X_nparray[:, -1, :]):
            print("Detected special info in first column, swapping...")
            X_nparray = np.concatenate((X_nparray[:, 1:, :], X_nparray[:, 0:1, :]), axis=1)
    
    X_flipped = np.flip(X_nparray, axis=2)  
    y_nparray = y_nparray.flatten()
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
        print(all_probs.shape)
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

def loadmat(filename):
    with h5py.File(filename, 'r') as data:
        if 'Whole_tracks' not in data:
            raise KeyError("❌ Error: 'Whole_tracks' doen't exist！")
        
        whole_tracks = data['Whole_tracks']
        if 'count' not in whole_tracks or 'data' not in whole_tracks:
            raise KeyError(f"❌ Error: 'Whole_tracks' incomplete, included: {list(whole_tracks.keys())}")

        # 读取 count
        count = int(whole_tracks['count'][()].item())
        track = [np.transpose(data[whole_tracks['data'][i].item()][:]).astype(np.float32) for i in range(count)]
    
    return {'tracks': {'count': count, 'data': track}}

def load_labels(label_path):
    with h5py.File(label_path, 'r') as data:
        if 'class_label' not in data:
            raise KeyError("❌ error: 'class_label' doen't exist")
        
        class_label = data['class_label'][()]
        
        if isinstance(class_label, np.ndarray):
            if class_label.size == 1:  
                class_label = class_label.item()
            else:  
                class_label = np.array(class_label)
        else:
            class_label = int(class_label)

        print(f"✅  class_label, shape: {class_label.shape}")
        return class_label
    
def process_file(matpath, label_path, model, roi_extractor, clustering_layer, device, NCLASS, args_test_batch_size):
    print(f"📌 data path: {matpath}")
    
    mat = loadmat(matpath)
    X_test = mat['tracks']['data']
    X_test = np.asarray(X_test).astype(np.float32)
    X_test_original = np.transpose(X_test, (0, 2, 1))

    y_test = load_labels(label_path)
    y_test_list = y_test

    X_test, y_test = udflip(X_test_original, y_test, shuffle=False)

    y_test = torch.from_numpy(y_test.astype(np.int64)).to(device)  
    X_test = torch.from_numpy(X_test).to(device)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    tst_set = utils.TensorDataset(X_test, y_test)
    tst_loader = utils.DataLoader(tst_set, batch_size=args_test_batch_size, shuffle=False, **kwargs)

    model.to(device)
    roi_extractor.to(device)
    clustering_layer.to(device)
    model.eval()
    roi_extractor.eval()
    clustering_layer.eval()

    probs, labels = [], []

    loss_nll = torch.nn.NLLLoss()
    with torch.no_grad():
        for data, target in tst_loader:
            labels += target.cpu().numpy().tolist()

            data, target = data.to(device), target.to(device)

            data_processed = preprocess_fiber_input(data, roi_extractor=roi_extractor, device=device, net_type='FE')

            output, embed, *_ = model(data_processed)  

            probs.append(output.data.cpu().numpy())  # 

    preds = aug_at_test(probs, mode='max')

    conf_mat = confusion_matrix(y_test_list, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_list, preds, average='macro')

    try:
        probs = np.concatenate(probs, axis=0)
        probs = F.softmax(torch.tensor(probs), dim=1).numpy()
        labels = np.array(labels)
        preds = np.argmax(probs, axis=1)

        auroc = roc_auc_score(labels, probs, multi_class='ovr')
        auprc = average_precision_score(labels, probs, average='macro')
    except ValueError as e:
        print(f"AUROC / AUPRC Error: {e}")
        auroc, auprc = None, None

    return precision, recall, f1, auroc, auprc  

def test(model, roi_extractor, tst_loader, device):
    model.eval()
    logit=list()
    attVec=list()
    for data,lbl in tst_loader:
        with torch.no_grad():  # error corrected by MH 10/12/2022 (add with torch.no_grad():) 
            data = Variable(data.cuda())
            data_processed = preprocess_fiber_input(data, roi_extractor=roi_extractor, device=device, net_type='FE')
            output,_,att,_, _, _, _, _, _, _, _ = model(data_processed)
            logit.append(output.data.cpu().numpy())
            attVec.append(att.data.cpu().numpy())
    return logit,attVec


matpath = '../Testing_Set/J0037_tracks.mat'
label_path = '../Testing_Set/J0037_class_label.mat'  
classnum = 15  
ROI_EMBEDDING_DIM = 32

args_test_batch_size = 10000
NCLASS = int(classnum)

print(f"📌 deal with data: {matpath}")
mat = loadmat(matpath)
X_test = mat['tracks']['data']
X_test = np.asarray(X_test).astype(np.float32)
X_test_original = np.transpose(X_test, (0, 2, 1))  

def udflip(X_nparray, y_nparray, shuffle=True):

    if X_nparray.shape[2] == 4:
        if np.std(X_nparray[:, 0, :]) > np.std(X_nparray[:, -1, :]):
            print("Detected special info in first column, swapping...")
            X_nparray = np.concatenate((X_nparray[:, 1:, :], X_nparray[:, 0:1, :]), axis=1)
    
    X_flipped = np.flip(X_nparray, axis=2)  
    y_nparray = y_nparray.flatten()
    X_aug = np.vstack((X_nparray, X_flipped))
    y_aug = np.hstack((y_nparray, y_nparray))  

    if shuffle:
        shuffle_idx = np.random.permutation(X_aug.shape[0])
        return X_aug[shuffle_idx], y_aug[shuffle_idx]
    else:
        return X_aug, y_aug
y_test = load_labels(label_path)
y_test_list = y_test
print(X_test_original.shape)
print(y_test.shape)
X_test, y_test = udflip(X_test_original,y_test,shuffle=False)
print(X_test.shape)
print(y_test.shape)
X_test_np = X_test.copy()
y_test = torch.from_numpy(y_test.astype(np.int64)) 
X_test = torch.from_numpy(X_test)

kwargs = {'num_workers': 1, 'pin_memory': True}
tst_set = utils.TensorDataset(X_test, y_test)
tst_loader = utils.DataLoader(tst_set, batch_size=args_test_batch_size, shuffle=False, **kwargs)




modelpath = 'save_smal/focal_loss_and_cluster_loss_c_10.0_FE_dim_32.model'
fe_path = 'save_smal/FE_layer_focal_loss_and_cluster_loss_c_10.0_FE_dim_32.model'
cls_path = 'save_smal/CLS_layer_focal_loss_and_cluster_loss_c_10.0_FE_dim_32.model'
# 加载模型
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
ROI_EMBEDDING_DIM = 32
NUM_ROI_CLASSES = 726 + 1
HIDDEN_DIM = 64
model=RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=3+ROI_EMBEDDING_DIM)
# init ROI Embedding layer
roi_embedding_layer = nn.Embedding(NUM_ROI_CLASSES, ROI_EMBEDDING_DIM).to(device)
# init FE
roi_extractor = ROIFeatureExtractor(roi_embedding_layer, ROI_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)
roi_extractor.to(device)
model.to(device)
clustering_layer = ClusterlingLayer(embedding_dimension=512, num_clusters=NCLASS, alpha=1.0)
kl_loss = KLDivLoss(NCLASS, loss_weight=2.0, temperature=2)
kl_loss.to(device)
clustering_layer.to(device)
# 2️⃣ 加载权重
state_dict = torch.load(modelpath, map_location=device)
state_dict_FE = torch.load(fe_path, map_location=device)
state_dict_cls = torch.load(cls_path, map_location=device)
model.load_state_dict(state_dict)
roi_extractor.load_state_dict(state_dict_FE)
clustering_layer.load_state_dict(state_dict_cls)
model.eval()
roi_extractor.eval()
clustering_layer.eval()

log_testing_total_loss = 0.0
log_focal_loss = 0.0
log_centering_loss= 0.
log_clustering_loss = 0.0
probs = []
preds = []
labels = []

global global_cluster_rois  # Ensure global access to cluster anatomical profiles
loss_nll = nn.NLLLoss(size_average=True) # log-softmax applied in the network
with torch.no_grad():
    logit, attVec = test(model, roi_extractor, tst_loader, device)
    attVec=mySoftmax(np.squeeze(np.vstack(attVec))).astype(np.float32)
    print('size of attVec',attVec.shape)
    #build output
    prob=np.exp(np.vstack(logit)).astype(np.float32)
    membership=np.argmax(prob,axis=1).reshape((-1,1)).astype(np.float32)
    maxprob=np.amax(prob,axis=1).reshape((-1,1)).astype(np.float32)
    
    output_max=np.zeros((X_test_np.shape[0],7),dtype=np.float32)
    output_max[:,0]=np.arange(1,X_test_np.shape[0]+1)
    for i in range(X_test_np.shape[0]):
        output_max[i, 0] = i + 1  # 
        output_max[i, 1:4] = X_test_np[i, 0:3, 0]  
        output_max[i, 4] = X_test_np[i, 3, 0]  
    #merge
    output_max=np.hstack((output_max,prob,membership,maxprob))
    np.savetxt(matpath.replace('.mat','.txt'),output_max,fmt='%.4e')
        # Compute clustering loss if enabled
    for i in range(NCLASS):
        #print(i)
        submat=output_max[np.where(output_max[:,-2]==i)]
        #fiber index
        fiberIndex=submat[:,0].reshape((-1,1))
        np.savetxt(matpath.replace('.mat','_'+'{0:02}'.format(i)+'_fiberindex.txt'),fiberIndex,fmt='%d')
        #fiber prob
        fiberProb=submat[:,-1].reshape((-1,1))
        np.savetxt(matpath.replace('.mat','_'+'{0:02}'.format(i)+'_fiberprob.txt'),fiberProb,fmt='%.4e')
        #fiber attention map
        fiberAtm=attVec[np.where(output_max[:,-2]==i)]
        #print('fiberAtm.shape',fiberAtm.shape)
        np.savetxt(matpath.replace('.mat','_'+'{0:02}'.format(i)+'_fiberatm.txt'),fiberAtm,fmt='%.4e')        
    print('results saved!')