import torch
import torch.nn.functional as F
A = torch.load('./result/network_output.pt')
import torch.nn as nn
def my_log_softmax(cls):
    cls = cls.unsqueeze(0)
    # N, 2, 25, 25
    cls = cls.view(1, 2, 1, 25, 25)
    # N, 2, 1, 25, 25
    cls = cls.permute(0, 2, 3, 4, 1).contiguous()
    # N, 1, 25, 25, 2
    cls = F.log_softmax(cls, dim=4) # The last dim, shrink the scores to range (0,1)
    return cls
criterion = nn.CrossEntropyLoss()
GT = A['GT_cls'][0][1].unsqueeze(-1)
score = A['cls'][0][1]
score = my_log_softmax(score)
score = score.view(-1, 2)
GT = GT.view(-1)
pos_inds = torch.nonzero(GT > 0).squeeze(1)
print(pos_inds)
print(criterion(score, GT))
# print(score[:, 1])