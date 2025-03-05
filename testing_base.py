import os
import sys
# del os.environ['MKL_NUM_THREADS'] # error corrected by MH 10/12/2022 (add these three lines)
import torch
from torch.autograd import Variable
import torch.utils.data as utils
import numpy as np
import gc
import sys
import scipy.io as spio
import h5py
import RESNET152_ATT_naive
import torch
import numpy as np
import h5py
import time
import gc
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch.utils.data as utils
# 数据加载
import os
import sys
# del os.environ['MKL_NUM_THREADS'] # error corrected by MH 10/12/2022 (add these three lines)
from Embedding_layer import ROIFeatureExtractor
import torch
from torch.autograd import Variable
import torch.utils.data as utils
import numpy as np
import gc
import sys
import scipy.io as spio
import RESNET152_ATT_naive
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch.nn as nn
from Util import focalLoss, preprocess_fiber_input
from clustering_layer_v2 import ClusterlingLayer
from klDiv import KLDivLoss
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
import argparse


import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as utils
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
import torch.nn.functional as F

def loadmat(filename):
    '''
    读取 MATLAB v7.3 `.mat` 文件（Whole_tracks 作为 tracks）
    '''
    output = dict()
    
    # 打开 HDF5 MAT 文件
    with h5py.File(filename, 'r') as data:
        # 读取 Whole_tracks 变量
        if 'Whole_tracks' not in data:
            raise KeyError("❌ 错误: 'Whole_tracks' 变量不存在！")

        whole_tracks = data['Whole_tracks']  # 结构体 Whole_tracks

        # 确保它有 `count` 和 `data`
        if 'count' not in whole_tracks or 'data' not in whole_tracks:
            raise KeyError(f"❌ 错误: 'Whole_tracks' 结构不完整！包含: {list(whole_tracks.keys())}")

        # 读取 count（可能是字符编码格式，需要解析）
        count = whole_tracks['count'][()]  
        print("🔍 Whole_tracks['count'] 数据:", count)
        print("🔍 数据类型:", type(count))

        # 直接转换成整数
        total_count = int(count.item())
        print(f'total_count: {total_count}')
        # 读取 Whole_tracks['data']
        track = []
        for i in range(total_count):
            data_ref = whole_tracks['data'][i].item()
            track.append(np.transpose(data[data_ref][:]).astype(np.float32))

        # 组织输出
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
    """ 读取 MATLAB v7.3 .mat 文件 """
    with h5py.File(filename, 'r') as data:
        if 'Whole_tracks' not in data:
            raise KeyError("❌ 错误: 'Whole_tracks' 变量不存在！")
        
        whole_tracks = data['Whole_tracks']
        if 'count' not in whole_tracks or 'data' not in whole_tracks:
            raise KeyError(f"❌ 错误: 'Whole_tracks' 结构不完整！包含: {list(whole_tracks.keys())}")

        # 读取 count
        count = int(whole_tracks['count'][()].item())
        track = [np.transpose(data[whole_tracks['data'][i].item()][:]).astype(np.float32) for i in range(count)]
    
    return {'tracks': {'count': count, 'data': track}}

def load_labels(label_path):
    """ 读取标签 .mat 文件 """
    with h5py.File(label_path, 'r') as data:
        if 'class_label' not in data:
            raise KeyError("❌ 错误: 'class_label' 变量不存在！")
        
        class_label = data['class_label'][()]
        
        if isinstance(class_label, np.ndarray):
            if class_label.size == 1:  
                class_label = class_label.item()
            else:  
                class_label = np.array(class_label)
        else:
            class_label = int(class_label)

        print(f"✅ data loaded class_label, shape: {class_label.shape}")
        return class_label

def process_file(matpath, label_path, model, roi_extractor, clustering_layer, device, NCLASS, args_test_batch_size):
    """ 处理单个测试文件，并返回其指标 """
    print(f"📌 preprocess data: {matpath}")
    
    mat = loadmat(matpath)
    X_test = mat['tracks']['data']
    X_test = np.asarray(X_test).astype(np.float32)
    X_test_original = np.transpose(X_test, (0, 2, 1))

    # 读取标签
    y_test = load_labels(label_path)
    y_test_list = y_test

    # 数据增强
    X_test, y_test = udflip(X_test_original, y_test, shuffle=False)

    # 转换为 PyTorch Tensor 并移动到相同设备
    y_test = torch.from_numpy(y_test.astype(np.int64)).to(device)  # 确保标签也在正确设备上
    X_test = torch.from_numpy(X_test).to(device)

    kwargs = {'num_workers': 0, 'pin_memory': False}
    tst_set = utils.TensorDataset(X_test, y_test)
    tst_loader = utils.DataLoader(tst_set, batch_size=args_test_batch_size, shuffle=False, **kwargs)

    # **确保模型和 layers 都在同一个设备**
    model.to(device)
    model.eval()

    probs, labels = [], []

    loss_nll = torch.nn.NLLLoss()
    with torch.no_grad():
        for data, target in tst_loader:
            labels += target.cpu().numpy().tolist()

            # 确保 data 和 target 都在同一设备
            data, target = data.to(device), target.to(device)

            # 预处理数据
            data_processed = preprocess_fiber_input(data, device=device, net_type='no_roi')

            # 送入模型
            output, embed, *_ = model(data_processed)  # **确保 model 已被移动到 `device`**

            probs.append(output.data.cpu().numpy())  # 确保 probs 存储在 CPU

    # 计算最终预测
    preds = aug_at_test(probs, mode='max')

    # 计算指标
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
        print(f"AUROC / AUPRC 计算错误: {e}")
        auroc, auprc = None, None

    return precision, recall, f1, auroc, auprc


def main():
    data_dir = '../Testing_Set/'  # 数据目录
    classnum = 15  # 类别数
    args_test_batch_size = 10000
    NCLASS = int(classnum)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=3)

    # 加载模型权重
    model.load_state_dict(torch.load('focal_loss_and_center_loss_no_roi.model', map_location=device))

    # 遍历所有 .mat 文件
    results = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_tracks.mat'):
            matpath = os.path.join(data_dir, filename)
            label_path = matpath.replace('_tracks.mat', '_class_label.mat')

            if not os.path.exists(label_path):
                print(f"❌ 找不到标签文件: {label_path}")
                continue
            start_time = time.time()
            precision, recall, f1, auroc, auprc = process_file(
                matpath, label_path, model, None, None, device, NCLASS, args_test_batch_size
            )
            print(time.time()-start_time,'seconds')
            print(f"📊 {filename} 指标:")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"  AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

            results.append([precision, recall, f1, auroc, auprc])

    # 计算均值和标准差
    results = np.array(results)
    mean_values = np.mean(results, axis=0)
    std_values = np.std(results, axis=0)

    print("\n📊 所有测试文件的平均指标:")
    print(f"  Precision: {mean_values[0]:.4f} ± {std_values[0]:.4f}")
    print(f"  Recall: {mean_values[1]:.4f} ± {std_values[1]:.4f}")
    print(f"  F1-score: {mean_values[2]:.4f} ± {std_values[2]:.4f}")
    print(f"  AUROC: {mean_values[3]:.4f} ± {std_values[3]:.4f}")
    print(f"  AUPRC: {mean_values[4]:.4f} ± {std_values[4]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='naive CNN with weighted loss')
    main()
