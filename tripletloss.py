import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #print(dist[0])
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(sorted(dist[i][mask[i]])[-3])
            dist_an.append(sorted(dist[i][mask[i] == 0])[5])
        dist_ap = torch.stack((dist_ap))
        dist_an = torch.stack((dist_an))
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() 
        return loss, prec
    
class TripletLoss_cosine(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss_cosine, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist = 1 - F.cosine_similarity(inputs[i].expand_as(inputs),inputs)
            dist_ap.append(sorted(dist[mask[i]])[-3])
            dist_an.append(sorted(dist[mask[i] == 0])[5])
            
            
        # Compute pairwise distance, replace by the official when merged
        
        dist_ap = torch.stack((dist_ap))
        dist_an = torch.stack((dist_an))
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() 
        return loss, prec