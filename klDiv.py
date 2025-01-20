# KL-Divergence Loss
import torch
import torch.nn as nn
from torch.autograd import Variable

class KLDivLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=1.0, temperature=1):
        super(KLDivLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, output, target_output):
        """Compute KL-Divergence Loss"""
        """
        param: output: middle ouptput logits/deep features.
        param: target_output: final output divided by temperature and softmax.
        """

        output = output / self.temperature
        output_log_softmax = torch.log_softmax(output, dim=1)

        loss_kd = nn.KLDivLoss()(output_log_softmax, target_output)
        # loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return self.loss_weight * loss_kd * (self.temperature**2)
    
    def kl_div_cluster(self, output, target_output):
        """Compute KL-Divergence Loss for clustering layer"""
        """
        param: output: clustering output
        param: target_output: target output distribution from ground-truth labels
        """
        loss_kd = nn.KLDivLoss(reduction='sum')(output, target_output)
        return loss_kd


    
