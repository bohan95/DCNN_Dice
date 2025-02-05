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
        Compute Dice Loss for each fiber against all clusters.

        Parameters:
            fiber_data: List[Tensor], 每个 fiber 的 ROI category，长度等于 batch_size。
            cluster_roi_true: List[Tensor], 每个 cluster 的 ROI category，长度等于 num_clusters。
            smooth: float, 避免除零的小数值。

        Returns:
            dice_losses: Tensor, shape (batch_size, num_clusters)，
                        每个 fiber 和每个 cluster 的 Dice Loss。
        """
        batch_size = len(fiber_data)  # Fiber 数量
        num_clusters = len(cluster_roi_true)  # Cluster 数量
        dice_losses = torch.zeros((batch_size, num_clusters))  # 结果存储张量

        for batch_idx in range(batch_size):  # 遍历 batch 里的每个 fiber
            ROI_fiber = fiber_data[batch_idx]  # 当前 fiber 的 ROI
            fiber_losses = []

            for cluster_idx in range(num_clusters):  # 遍历所有 clusters
                ROI_cluster = cluster_roi_true[cluster_idx]  # 当前 cluster 的 ROI

                if ROI_fiber.numel() == 0 and ROI_cluster.numel() == 0:
                    dice_loss = torch.tensor(0.0)  # 两者都是空集，Dice Loss = 0
                elif ROI_fiber.numel() == 0 or ROI_cluster.numel() == 0:
                    dice_loss = torch.tensor(1.0)  # 其中一个为空集，Dice Loss = 1
                else:
                    # 计算 Dice Loss
                    intersection = (ROI_fiber.unsqueeze(1) == ROI_cluster.unsqueeze(0)).sum().float()
                    dice_score = (2. * intersection + smooth) / (ROI_fiber.numel() + ROI_cluster.numel() + smooth)
                    dice_loss = 1 - dice_score  # 1 - Dice Score = Dice Loss

                fiber_losses.append(dice_loss)  # 存储当前 fiber 和 cluster 的 loss

            dice_losses[batch_idx, :] = torch.stack(fiber_losses)  # 存入 batch 结果

        return dice_losses  # Shape: (batch_size, num_clusters)



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
