import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterlingLayer(nn.Module):
    def __init__(self, embedding_dimension=512, num_clusters=2, alpha=1.0, lambda_dice=1.0):
        super(ClusterlingLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.lambda_dice = lambda_dice  # for Dice Loss
        self.weight = nn.Parameter(torch.Tensor(self.num_clusters, self.embedding_dimension))
        self.weight = nn.init.xavier_uniform_(self.weight)  # Xavier init

    def forward(self, x, anatomical_info=None, cluster_rois=None, predic=None):
        """Calculate clustering probability, combined with anatomical consistency Dice Loss"""
        x = x.unsqueeze(1) - self.weight  # cal dis
        x = torch.mul(x, x).sum(dim=2)
        x_dis = x.clone()

        if anatomical_info is not None and cluster_rois is not None:
            dice_loss_anatomical = self.dice_loss_fiber(anatomical_info, cluster_rois).to(x.device)
            # TODO we can try this later 
            soft_label_adjustment = torch.exp(-self.lambda_dice * dice_loss_anatomical).to(x.device)     
        else:
            dice_loss_anatomical = 1.0
            soft_label_adjustment = 1.0

        x = x * dice_loss_anatomical  #  Dice Loss 

        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)

        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)

        return x, x_dis  

    def dice_loss_fiber(self, fiber_data, cluster_roi_true, smooth=1e-6):
        """
            Calculate the Dice Loss of fiber and cluster in batch, vectorized implementation
            fiberdata: List[Tensor], the ROI classification of each fiber
            clusterroi_true: List[Tensor], the ROI classification of each cluster
        """
        batch_size = len(fiber_data)
        num_clusters = len(cluster_roi_true)

        # get fiber's one-hot (batch_size, num_anatomical_rois)
        fiber_rois_onehot = torch.zeros((batch_size, 726 + 1), device=fiber_data[0].device)
        for i, roi in enumerate(fiber_data):
            roi = roi.long()  # make sure index is long 
            fiber_rois_onehot[i, roi] = 1  

        # get cluster's one-hot (num_clusters, num_anatomical_rois)
        cluster_rois_onehot = torch.zeros((num_clusters, 726 + 1), device=fiber_data[0].device)
        for i, roi in enumerate(cluster_roi_true):
            roi = roi.long()  # make sure index is long 
            cluster_rois_onehot[i, roi] = 1  

        # cal intersection(fiber and cluster ROI Intersection)
        intersection = (fiber_rois_onehot.unsqueeze(1) * cluster_rois_onehot.unsqueeze(0)).sum(dim=2).float()

        # cal Dice Loss
        fiber_size = fiber_rois_onehot.sum(dim=1, keepdim=True)
        cluster_size = cluster_rois_onehot.sum(dim=1, keepdim=True).T
        dice_score = (2.0 * intersection + smooth) / (fiber_size + cluster_size + smooth)

        return 1 - dice_score  # (batch_size, num_clusters)



    @staticmethod
    def target_distribution(batch: torch.Tensor) -> torch.Tensor:
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    @staticmethod
    def create_soft_labels(labels, num_classes, temperature=1.0):
        device = labels.device
        one_hot = torch.eye(num_classes, device=device)[labels.long()]
        soft_labels = F.softmax(one_hot / temperature, dim=1)
        return soft_labels

    def extra_repr(self):
        return f'embedding_dimension={self.embedding_dimension}, num_clusters={self.num_clusters}, alpha={self.alpha}, lambda_dice={self.lambda_dice}'

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
