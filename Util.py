import torch
import numpy as np
import torch.nn as nn

def preprocess_fiber_input(data, roi_extractor=None, roi_embedding_layer=None, device='cpu', net_type='FE'):
    if net_type == 'FE':
        """
        Process Fiber data, using 3D coordinates and ROI feature stitching as model input
        """
        coord_3d = data[:, :3, :].to(device)  # 3D (b, 3, 100)
        roi_indices = data[:, 3, :].long().to(device)  # ROI index (b, 100)

        # get ROI features
        roi_features = roi_extractor(roi_indices)  # (b, 100, embedding_dim)

        # Re-adjust the ROI dimensions to splice
        roi_features = roi_features.permute(0, 2, 1)  # (b, embedding_dim, 100)

        # Splicing 3D coordinates and ROI features
        processed_data = torch.cat([coord_3d, roi_features], dim=1)  # (b, 3 + embedding_dim, 100)

    elif net_type == 'EB':
        coord_3d = data[:, :3, :].to(device)  
        roi_indices = data[:, 3, :].long().to(roi_embedding_layer.weight.device)  
        roi_features = roi_embedding_layer(roi_indices)  
        roi_features = roi_features.permute(0, 2, 1)
        processed_data = torch.cat([coord_3d, roi_features], dim=1) 
    
    elif net_type == 'concat':
        processed_data = data
    else :
        processed_data = data[:, :3, :].to(device)
    
    return processed_data


"""v flip and shuffle"""
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
    - threshold: filter ROI threshold(default 40%)

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

def focalLoss(output,target, loss_nll):
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