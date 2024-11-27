import torch
import torch.nn as nn
import torch.nn.functional as F


# k-means clustering loss
class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()
        
    def forward(self, inner_products):
        cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), inner_products.shape[1]).permute(0, 3, 1, 2).to(torch.float64)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        return cluster_loss, cluster_probs
    
    
# cosine similarity clustering layer 
class ConvClusterProbe(nn.Module):
    def __init__(self, in_channels, out_channels, not_norm=False):
        super(ConvClusterProbe, self).__init__()
        print('-- Clusterer init')
        self.not_norm = not_norm
        self.cluster_centers = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, (1, 1),  bias=False), name='weight', dim=0)

    def forward(self, x): 
        if not self.not_norm:
            x = F.normalize(x, dim=1)
            self.cluster_centers.weight_g = torch.nn.Parameter(torch.ones_like(self.cluster_centers.weight_g))
        return self.cluster_centers(x)