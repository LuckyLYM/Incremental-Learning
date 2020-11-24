from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable
# python debugger
import pdb


# I need some time to digest this part. seems difficulty to understand

class TripletLossNoHardMining(nn.Module):
    def __init__(self, margin=0, num_instances=8):
        super(TripletLossNoHardMining, self).__init__()
        self.margin = margin
        self.num_instances = num_instances
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise Euclidean distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        # for numerical stability, smaller value is replaced by the given min value
        dist = dist.clamp(min=1e-12).sqrt()  


        # For each anchor, find the hardest positive and negative
        # mask[i][j] = 1, if example i and j have the same label
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())


        # no mining of the hard triplets
        dist_ap, dist_an = [], []

        # confused about this part, what if the index is out of boundary??
        for i in range(n):
            # the number of triplet pair for an anchor point
            pos_tmp = dist[i][mask[i]]
            neg_tmp = dist[i][mask[i] == 0]

            n_pair=min(len(pos_tmp),len(neg_tmp))
            n_pair=min(n_pair,self.num_instances)

            for j in range(n_pair):
                dist_ap.append(pos_tmp[j])
                dist_an.append(neg_tmp[j])

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)


        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_p = torch.mean(dist_ap).data.item()
        dist_n = torch.mean(dist_an).data.item()
        return loss, prec, dist_p, dist_n
