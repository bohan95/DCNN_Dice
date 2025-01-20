# Clustering layer for the clustering loss
import torch
import torch.nn as nn
from torch.autograd import Variable
from klDiv import KLDivLoss
import torch.nn.functional as F


class ClusterlingLayer(nn.Module):
    def __init__(self, embedding_dimension=512, num_clusters=2, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.num_clusters, self.embedding_dimension))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Compute Clustering Probability"""
        """
        param: x: output of deep ResNet network
        x_shape: (batch_size, embedding_dimension)
        """
        # (batch_size, 1, embedding_dimension) -> (batch_size, num_clusters, embedding_dimension)
        x = x.unsqueeze(1) - self.weight 
        
        # computing the squared Euclidean distance
        x = torch.mul(x, x)
        # sums the squared differences over the embedding_dimension dimension, 
        # giving a shape of (batch_size, num_clusters)
        x = torch.sum(x, dim=2)
        # x_dis stores the original computed distances
        x_dis = x

        # The distances are transformed using the parameter alpha to modulate the distances. 
        # This sequence of operations ensures that the distances are scaled and inverted appropriately
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)

        # The distances are normalized to sum to 1 over the num_clusters dimension
        # This step ensures that the resulting values are probabilities that sum to 1 for each sample.
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)

        return x, x_dis # x_dis is the original computed distances; x is transformed distances
    
    @staticmethod
    def target_distribution(batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in Equation 3 of
        Xie/Girshick/Farhadi; this is used in the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    
    # function to create soft labels from clustering loss 
    @staticmethod
    def create_soft_labels(labels, num_classes, temperature=1.0):
        """
        Create soft labels using temperature scaling.
        """
        one_hot = torch.eye(num_classes)[labels.long()]
        # Apply temperature scaling
        soft_labels = F.softmax(one_hot / temperature, dim=1)
        return soft_labels
    
    def extra_repr(self):
        return 'embedding_dimension(in_features)={}, num_clusters(out_features)={}, alpha={}'.format(
            self.embedding_dimension, self.num_clusters, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)

    