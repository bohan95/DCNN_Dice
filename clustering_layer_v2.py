import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterlingLayer(nn.Module):
    def __init__(self, embedding_dimension=512, num_clusters=2, alpha=1.0, lambda_dice=1.0):
        super(ClusterlingLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.lambda_dice = lambda_dice  # 控制 Dice Loss 影响力的超参数
        self.weight = nn.Parameter(torch.Tensor(self.num_clusters, self.embedding_dimension))
        self.weight = nn.init.xavier_uniform_(self.weight)  # Xavier 初始化

    def forward(self, x, anatomical_info=None, cluster_rois=None):
        """计算聚类概率，结合解剖一致性 Dice Loss"""
        x = x.unsqueeze(1) - self.weight  # 计算欧几里得距离
        x = torch.mul(x, x).sum(dim=2)
        x_dis = x.clone()

        if anatomical_info is not None and cluster_rois is not None:
            dice_loss_anatomical = self.dice_loss_fiber(anatomical_info, cluster_rois, threshold=0.4)
            soft_label_adjustment = torch.exp(-self.lambda_dice * dice_loss_anatomical)
        else:
            soft_label_adjustment = 1.0

        print(f'x before:{x}')
        x = x * soft_label_adjustment  # 结合 Dice Loss 计算最终距离
        print(f'x after:{x}')
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)

        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)

        return x, x_dis  

    def dice_loss_fiber(fiber_data, fiber_pred, cluster_roi_true, smooth=1e-6):
        """
        Compute the fiber-level ROI Dice Loss based only on the predicted cluster.

        Parameters:
            fiber_data: Tensor, shape (b, 4, 100), containing fiber ROI information.
            fiber_pred: Tensor, shape (b,), model-predicted fiber cluster labels.
            cluster_roi_true: List[Tensor], ground-truth ROI categories for each cluster (each cluster has one tensor).
            smooth: float, a small value to avoid division by zero.

        Returns:
            dice_losses_per_batch: Tensor, shape (b,), containing Dice Loss for each batch element.
        """
        fiber_rois = fiber_data  # Compute ROI categories for each fiber
        batch_size = fiber_data.shape[0]  # Get batch size
        fiber_losses_per_batch = []

        for batch_idx in range(batch_size):  # Iterate over each batch
            ROI_fiber = fiber_rois[batch_idx]  # Fiber-level ROI
            pred_cluster = fiber_pred[batch_idx].item()  # Model-predicted cluster
            
            # Get predicted cluster-level ROI
            ROI_cluster_pred = cluster_roi_true[pred_cluster] if pred_cluster < len(cluster_roi_true) else torch.tensor([])

            if ROI_fiber.numel() == 0 and ROI_cluster_pred.numel() == 0:
                fiber_losses_per_batch.append(torch.tensor(0.0))  # If both are empty, loss = 0
            elif ROI_fiber.numel() == 0 or ROI_cluster_pred.numel() == 0:
                fiber_losses_per_batch.append(torch.tensor(1.0))  # If only one is empty, loss = 1
            else:
                # Compute Dice Loss based on fiber ROI and predicted cluster ROI
                intersection = (ROI_fiber.unsqueeze(1) == ROI_cluster_pred.unsqueeze(0)).sum().float()
                dice_score = (2. * intersection + smooth) / (ROI_fiber.numel() + ROI_cluster_pred.numel() + smooth)
                fiber_losses_per_batch.append(1 - dice_score)  # 1 - Dice coefficient = Dice Loss

        return torch.tensor(fiber_losses_per_batch)  # Shape: (b,)

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
