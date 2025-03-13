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

from Util import *

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
            data_processed = preprocess_fiber_input(data, device=device, net_type='concat')

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
    ROI_EMBEDDING_DIM = 32
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=4)

    # 加载模型权重
    model.load_state_dict(torch.load('focal_loss_and_cluster_loss_c_10.0_concat.model', map_location=device))

    # 遍历所有 .mat 文件
    results = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_tracks.mat'):
            matpath = os.path.join(data_dir, filename)
            label_path = matpath.replace('_tracks.mat', '_class_label.mat')

            if not os.path.exists(label_path):
                print(f"❌ Label file not found: {label_path}")
                continue
            start_time = time.time()
            precision, recall, f1, auroc, auprc = process_file(
                matpath, label_path, model, None, None, device, NCLASS, args_test_batch_size
            )
            print(time.time()-start_time,'seconds')
            print(f"📊 {filename} results:")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"  AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

            results.append([precision, recall, f1, auroc, auprc])

    # 计算均值和标准差
    results = np.array(results)
    mean_values = np.mean(results, axis=0)
    std_values = np.std(results, axis=0)

    print("\n📊 Average metrics for all test files:")
    print(f"  Precision: {mean_values[0]:.4f} ± {std_values[0]:.4f}")
    print(f"  Recall: {mean_values[1]:.4f} ± {std_values[1]:.4f}")
    print(f"  F1-score: {mean_values[2]:.4f} ± {std_values[2]:.4f}")
    print(f"  AUROC: {mean_values[3]:.4f} ± {std_values[3]:.4f}")
    print(f"  AUPRC: {mean_values[4]:.4f} ± {std_values[4]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='naive CNN with weighted loss')
    main()
