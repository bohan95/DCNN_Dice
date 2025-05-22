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


# Custom module imports
from Embedding_layer import ROIFeatureExtractor
import RESNET152_ATT_naive
from Util import preprocess_fiber_input
from clustering_layer_v2 import ClusterlingLayer
from klDiv import KLDivLoss
       
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

def datato3d(arrays):#list of np arrays, NULL*3*100
    output=list()
    for i in arrays:
        i=np.squeeze(i,axis=1)
        i=np.transpose(i,(0,2,1))
        output.append(i)
    return output
      

def loadmat(filename):
    with h5py.File(filename, 'r') as data:
        if 'Whole_tracks' not in data:
            raise KeyError("❌ Error: 'Whole_tracks' doen't exist！")
        
        whole_tracks = data['Whole_tracks']
        if 'count' not in whole_tracks or 'data' not in whole_tracks:
            raise KeyError(f"❌ Error: 'Whole_tracks' incomplete, included: {list(whole_tracks.keys())}")

        count = int(whole_tracks['count'][()].item())
        track = [np.transpose(data[whole_tracks['data'][i].item()][:]).astype(np.float32) for i in range(count)]
    
    return {'tracks': {'count': count, 'data': track}}
  
def test(model, roi_extractor, tst_loader, device):
    model.eval()
    logit=list()
    attVec=list()
    for data in tst_loader:
        with torch.no_grad():  # error corrected by MH 10/12/2022 (add with torch.no_grad():) 
            # print(f"data type: {type(data)}")
            # print(data)
            data = data[0]
            data = Variable(data.cuda())
            data_processed = preprocess_fiber_input(data, roi_extractor=roi_extractor, device=device, net_type='FE')
            output,_,att,_, _, _, _, _, _, _, _ = model(data_processed)
            logit.append(output.data.cpu().numpy())
            attVec.append(att.data.cpu().numpy())
    return logit,attVec

# TODO Please change the file name
matpath = '../Testing_Set/J0037_tracks.mat'
# label_path = '../Testing_Set/J0037_class_label.mat'  
classnum = 15  
ROI_EMBEDDING_DIM = 32

args_test_batch_size = 10000
NCLASS = int(classnum)

print(f"📌 deal with data: {matpath}")
mat = loadmat(matpath)
X_test = mat['tracks']['data']
X_test = np.asarray(X_test).astype(np.float32)
X_test_original = np.transpose(X_test, (0, 2, 1))  

def udflip_x_only(X_nparray, shuffle=True):

    if X_nparray.shape[2] == 4:
        if np.std(X_nparray[:, 0, :]) > np.std(X_nparray[:, -1, :]):
            print("Detected special info in first column, swapping...")
            X_nparray = np.concatenate((X_nparray[:, 1:, :], X_nparray[:, 0:1, :]), axis=1)
    
    X_flipped = np.flip(X_nparray, axis=2)  
    X_aug = np.vstack((X_nparray, X_flipped))

    if shuffle:
        shuffle_idx = np.random.permutation(X_aug.shape[0])
        return X_aug[shuffle_idx]
    else:
        return X_aug
    
print(X_test_original.shape)
X_test = X_test_original
print(X_test.shape)
X_test_np = X_test.copy()
X_test = torch.from_numpy(X_test)
kwargs = {'num_workers': 1, 'pin_memory': True}
tst_set = utils.TensorDataset(X_test)
tst_loader = utils.DataLoader(tst_set, batch_size=args_test_batch_size, shuffle=False, **kwargs)


# TODO Please change the path 
modelpath = 'save_small/focal_loss_and_cluster_loss_c_10.0_FE_dim_32.model'
fe_path = 'save_small/FE_layer_focal_loss_and_cluster_loss_c_10.0_FE_dim_32.model'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

state_dict = torch.load(modelpath, map_location=device)
state_dict_FE = torch.load(fe_path, map_location=device)

model.load_state_dict(state_dict)
roi_extractor.load_state_dict(state_dict_FE)

model.eval()
roi_extractor.eval()

with torch.no_grad():
    logit, attVec = test(model, roi_extractor, tst_loader, device)
    attVec=mySoftmax(np.squeeze(np.vstack(attVec))).astype(np.float32)
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