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

    def forward(self, x, anatomical_info=None, cluster_rois=None, predic=None):
        """计算聚类概率，结合解剖一致性 Dice Loss"""
        x = x.unsqueeze(1) - self.weight  # 计算欧几里得距离
        x = torch.mul(x, x).sum(dim=2)
        x_dis = x.clone()

        if anatomical_info is not None and cluster_rois is not None:
            # print(f'anatomical_info: {anatomical_info.shape}')
            # print(f'anatomical_info: {predic.shape}')
            # print(f'anatomical_info: {cluster_rois.shape}')
            dice_loss_anatomical = self.dice_loss_fiber(anatomical_info, cluster_rois).to(x.device)
            soft_label_adjustment = torch.exp(-self.lambda_dice * dice_loss_anatomical).to(x.device)     
        else:
            dice_loss_anatomical = 1.0
            soft_label_adjustment = 1.0

        # print(f'x before:{x}')
        # print(f'dice_loss_anatomical:{dice_loss_anatomical}')
        # print(f'soft_label_adjustment:{soft_label_adjustment}')
        x = x * dice_loss_anatomical  # 结合 Dice Loss 计算最终距离
        # x = x * soft_label_adjustment  # 结合 Dice Loss 计算最终距离
        # print(f'x after:{x}')
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)

        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)

        return x, x_dis  

    def dice_loss_fiber(self, fiber_data, cluster_roi_true, smooth=1e-6):
        """
        计算 batch 里的 fiber 和 cluster 的 Dice Loss，向量化实现
        fiber_data: List[Tensor]，每个 fiber 的 ROI 分类
        cluster_roi_true: List[Tensor]，每个 cluster 的 ROI 分类
        """
        batch_size = len(fiber_data)
        num_clusters = len(cluster_roi_true)

        # 计算 fiber 的 one-hot 形式 (batch_size, num_anatomical_rois)
        fiber_rois_onehot = torch.zeros((batch_size, 726), device=fiber_data[0].device)
        # print(fiber_data)
        for i, roi in enumerate(fiber_data):
            roi = roi.long()  # 确保索引是 long 类型
            fiber_rois_onehot[i, roi] = 1  

        # 计算 cluster 的 one-hot 形式 (num_clusters, num_anatomical_rois)
        cluster_rois_onehot = torch.zeros((num_clusters, 726), device=fiber_data[0].device)
        for i, roi in enumerate(cluster_roi_true):
            roi = roi.long()  # 确保索引是 long 类型
            cluster_rois_onehot[i, roi] = 1  

        # 计算 intersection（fiber 和 cluster ROI 交集）
        intersection = (fiber_rois_onehot.unsqueeze(1) * cluster_rois_onehot.unsqueeze(0)).sum(dim=2).float()

        # 计算 Dice Loss
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
